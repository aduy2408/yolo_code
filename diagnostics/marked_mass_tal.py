#!/usr/bin/env python3
"""Measure per-GT responsibility assigned by YOLOv8 TaskAlignedAssigner.

This is an inference-only diagnostic. It never calls train() and never changes
model parameters. Dataset image symlinks in older LEVIR exports may point at a
stale absolute prefix, so --source-images is used as a basename fallback while
labels remain tied to the requested split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def local_ultralytics() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    package = ROOT / "models_related/ultralytics"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))


def read_boxes(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    classes = []
    if path.is_file():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) < 5:
                raise ValueError(f"{path}:{line_number}: expected class + xywh")
            cls, xc, yc, bw, bh = map(float, values[:5])
            if int(cls) < 0:
                raise ValueError(f"{path}:{line_number}: invalid class id")
            classes.append(int(cls))
            rows.append(((xc - bw / 2) * width, (yc - bh / 2) * height,
                         (xc + bw / 2) * width, (yc + bh / 2) * height))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2, (size - new_height) / 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, ratio, float(left), float(top)


def transform_boxes(boxes: np.ndarray, ratio: float, pad_x: float, pad_y: float) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    transformed = boxes.copy()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * ratio + pad_x
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * ratio + pad_y
    return transformed


def image_for(path: Path, source_images: Path) -> tuple[Path, np.ndarray]:
    candidates = [path, source_images / path.name]
    for candidate in candidates:
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is not None:
            return candidate, image
    raise FileNotFoundError(f"Could not read {path} or basename fallback under {source_images}")


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    tl = np.maximum(box[:2], boxes[:, :2])
    br = np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.maximum(br - tl, 0), axis=1)
    area_a = max(float(np.prod(box[2:] - box[:2])), 0.0)
    area_b = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = lambda key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    area = numeric("bbox_area")
    mass = numeric("sum_target_score")
    density = numeric("local_gt_count")
    miss = 1.0 - numeric("matched_at_eval")
    low = mass <= np.quantile(mass, 0.25) if len(mass) else np.zeros(0, dtype=bool)
    tiny = area <= np.quantile(area, 0.25) if len(area) else np.zeros(0, dtype=bool)
    return {
        "objects": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "mean_recall_iou50": float(1.0 - miss.mean()) if len(miss) else None,
        "spearman_size_vs_num_positive": rank_correlation(numeric("bbox_area"), numeric("num_assigned")),
        "spearman_size_vs_sum_positive_weight": rank_correlation(area, mass),
        "spearman_density_vs_sum_positive_weight": rank_correlation(density, mass),
        "spearman_sum_positive_weight_vs_miss": rank_correlation(mass, miss),
        "tiny_q25_mean_sum_positive_weight": float(mass[tiny].mean()) if tiny.any() else None,
        "non_tiny_q75_mean_sum_positive_weight": float(mass[~tiny].mean()) if (~tiny).any() else None,
        "lowest_mass_quartile_miss_rate": float(miss[low].mean()) if low.any() else None,
        "highest_mass_quartile_miss_rate": float(miss[~low].mean()) if (~low).any() else None,
        "marked_mass_signal_rule": (
            "signal_candidate" if len(rows) and tiny.any() and low.any()
            and float(mass[tiny].mean()) < float(mass[~tiny].mean())
            and float(miss[low].mean()) > float(miss[~low].mean())
            else "no_clear_signal"
        ),
    }


def plot_rows(rows: list[dict[str, object]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x = np.asarray([float(r["bbox_area"]) for r in rows])
    m = np.asarray([float(r["sum_target_score"]) for r in rows])
    n = np.asarray([float(r["num_assigned"]) for r in rows])
    d = np.asarray([float(r["local_gt_count"]) for r in rows])
    miss = np.asarray([float(r["matched_at_eval"]) == 0 for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].scatter(x, n, c=miss, cmap="coolwarm", s=7, alpha=0.45)
    axes[0, 0].set(xscale="log", xlabel="GT area (input px²)", ylabel="num positive locations")
    axes[0, 1].scatter(x, m, c=miss, cmap="coolwarm", s=7, alpha=0.45)
    axes[0, 1].set(xscale="log", xlabel="GT area (input px²)", ylabel="sum target weight")
    axes[1, 0].scatter(m, miss.astype(float), s=7, alpha=0.35)
    axes[1, 0].set(xscale="log", xlabel="sum target weight", ylabel="miss at IoU 0.5")
    axes[1, 1].scatter(d, m, c=miss, cmap="coolwarm", s=7, alpha=0.45)
    axes[1, 1].set(xlabel="nearby GT count", ylabel="sum target weight")
    fig.suptitle("YOLOv8 TAL marked-mass diagnostic")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner

    model = YOLO(str(args.weights))
    net = model.model
    net.eval()
    head = net.model[-1]
    if not hasattr(head, "box_detail"):
        head.box_detail = torch.nn.ModuleList([torch.nn.Identity() for _ in range(head.nl)])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    net.to(device)
    strides = head.stride.detach().float().tolist()
    criterion = v8DetectionLoss(net)
    assigner = TaskAlignedAssigner(
        topk=int(getattr(net.args, "tal_topk", 10)),
        num_classes=int(head.nc),
        alpha=float(getattr(net.args, "tal_alpha", 0.5)),
        beta=float(getattr(net.args, "tal_beta", 6.0)),
        stride=strides,
        topk2=getattr(net.args, "tal_topk2", None),
    )
    image_paths = sorted((args.images).glob("*"))
    if args.limit:
        image_paths = image_paths[:args.limit]
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        actual_path, bgr = image_for(image_path, args.source_images)
        height, width = bgr.shape[:2]
        label_path = args.labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            label_path = args.source_labels / f"{image_path.stem}.txt"
        gt, gt_classes = read_boxes(label_path, width, height)
        model_image, ratio, pad_x, pad_y = letterbox(bgr, args.imgsz)
        gt = transform_boxes(gt, ratio, pad_x, pad_y)
        tensor = torch.from_numpy(cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
        with torch.inference_mode():
            inference, raw = net(tensor)
            if isinstance(raw, tuple):
                raw = raw[0]
            from ultralytics.utils.tal import make_anchors
            anchor_points, stride_tensor = make_anchors(raw["feats"], head.stride, 0.5)
            scores = raw["scores"].permute(0, 2, 1).contiguous()
            pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
            boxes = criterion.bbox_decode(anchor_points, pred_distri, None, stride_tensor)
            labels = torch.zeros((1, max(len(gt), 1), 1), dtype=torch.long, device=device)
            gt_xyxy = torch.zeros((1, max(len(gt), 1), 4), dtype=torch.float32, device=device)
            if len(gt):
                labels[0, :len(gt), 0] = torch.from_numpy(gt_classes).to(device)
                gt_xyxy[0, :len(gt)] = torch.from_numpy(gt).to(device)
            mask_gt = gt_xyxy.sum(2, keepdim=True).gt_(0.0)
            target_labels, target_boxes, target_scores, fg_mask, target_gt_idx = assigner(
                scores.sigmoid(), boxes * stride_tensor, anchor_points * stride_tensor, labels, gt_xyxy, mask_gt
            )
            detections = non_max_suppression(inference, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det)[0]
            det_boxes = detections[:, :4].detach().cpu().numpy() if len(detections) else np.empty((0, 4), dtype=np.float32)
            det_scores = detections[:, 4].detach().cpu().numpy() if len(detections) else np.empty(0, dtype=np.float32)
            assigned_idx = target_gt_idx[0].detach().cpu().numpy()
            foreground = fg_mask[0].bool().detach().cpu().numpy()
            target_weight = target_scores[0].detach().cpu().numpy()
        centers = (gt[:, :2] + gt[:, 2:]) / 2 if len(gt) else np.empty((0, 2))
        for gt_id, box in enumerate(gt):
            positive = foreground & (assigned_idx == gt_id)
            weights = target_weight[positive, int(gt_classes[gt_id])]
            distances = np.linalg.norm(centers - centers[gt_id], axis=1) if len(gt) else np.empty(0)
            nearby = int(((distances <= args.density_radius) & (distances > 0)).sum())
            overlaps = iou_one(box, det_boxes)
            best = int(overlaps.argmax()) if len(overlaps) else -1
            best_iou = float(overlaps[best]) if best >= 0 else 0.0
            rows.append({
                "image_id": image_path.stem,
                "image_path": str(actual_path),
                "gt_id": gt_id,
                "class_id": int(gt_classes[gt_id]),
                "bbox_w": float(box[2] - box[0]),
                "bbox_h": float(box[3] - box[1]),
                "bbox_area": float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0)),
                "num_assigned": int(positive.sum()),
                "sum_target_score": float(weights.sum()) if len(weights) else 0.0,
                "mean_target_score": float(weights.mean()) if len(weights) else 0.0,
                "max_target_score": float(weights.max()) if len(weights) else 0.0,
                "local_gt_count": nearby,
                "local_gt_density": float(nearby / (np.pi * args.density_radius ** 2)),
                "matched_at_eval": int(best_iou >= args.match_iou),
                "pred_score": float(det_scores[best]) if best >= 0 else 0.0,
                "pred_iou": best_iou,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["image_id"]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "weights": str(args.weights),
        "images_split": str(args.images),
        "labels_split": str(args.labels),
        "source_images": str(args.source_images),
        "imgsz": int(args.imgsz),
        "device": str(device),
        "tal": {"topk": assigner.topk, "alpha": assigner.alpha, "beta": assigner.beta, "strides": strides},
        "match_iou": args.match_iou,
        "density_radius": args.density_radius,
        "diagnostic_only": True,
        "analysis": analyse(rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_rows(rows, args.output.with_suffix(".png"))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Requested split image directory")
    parser.add_argument("--labels", type=Path, required=True, help="Requested split label directory")
    parser.add_argument("--source-images", type=Path, required=True, help="Canonical source image directory")
    parser.add_argument("--source-labels", type=Path, required=True, help="Canonical source label directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--density-radius", type=float, default=64.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
