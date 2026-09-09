#!/usr/bin/env python3
"""Diagnose TAL regression support by object size, geometry, raw loss, and failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.loss import bbox2dist, make_anchors, v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import bbox_iou, box_iou  # noqa: E402


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def size_group(area: float) -> str:
    if area < 100:
        return "tiny"
    if area <= 400:
        return "small"
    if area <= 1024:
        return "medium"
    return "large"


def read_labels(image: Path) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    rows = [line.split() for line in labels_for(image).read_text().splitlines() if line.strip()]
    cls = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32).reshape(-1, 1)
    boxes = torch.tensor(
        [[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32
    ).reshape(-1, 4)
    original = cv2.imread(str(image))
    if original is None:
        raise RuntimeError(f"could not read image: {image}")
    height, width = original.shape[:2]
    areas = [float(box[2] * width * box[3] * height) for box in boxes]
    return cls, boxes, areas


def finite_corr(a: list[float], b: list[float]) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def nwd_similarity(boxes: torch.Tensor, gt: torch.Tensor, c: float) -> torch.Tensor:
    """Normalized Gaussian Wasserstein similarity for xyxy boxes in pixels."""
    pred_xywh = torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]), dim=1
    )
    gt_xywh = torch.stack(((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2, gt[2] - gt[0], gt[3] - gt[1]))
    w2 = (pred_xywh[:, :2] - gt_xywh[:2]).square().sum(1)
    w2 = w2 + 0.25 * (pred_xywh[:, 2:] - gt_xywh[2:]).square().sum(1)
    return torch.exp(-torch.sqrt(w2.clamp_min(0)) / max(c, 1e-6))


def actual_matches(wrapper: YOLO, image: Path, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = wrapper.predict(
        source=str(image), imgsz=args.imgsz, device=args.device, conf=args.conf,
        verbose=False, augment=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    return (
        result.boxes.xyxy.detach().float().cpu(),
        result.boxes.conf.detach().float().cpu(),
        result.boxes.cls.detach().long().cpu(),
    )


def inspect_checkpoint(checkpoint: Path, images: list[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if args.expected_seed is not None and train_args.get("seed") != args.expected_seed:
        raise RuntimeError(f"expected checkpoint seed {args.expected_seed}, got {train_args.get('seed')!r}")
    net = wrapper.model.to(args.torch_device).eval()
    head = net.model[-1]
    strides = [float(value) for value in head.stride]
    if not strides or strides[0] != 4.0:
        raise RuntimeError(f"expected P2 stride first, got {strides}")

    loss = v8DetectionLoss(net)
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    rows: list[dict[str, object]] = []
    for image_index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.torch_device).permute(2, 0, 1).float()[None] / 255
        cls, boxes, areas = read_labels(image_path)
        if len(boxes) == 0:
            continue
        batch = {
            "batch_idx": torch.zeros((len(boxes),), dtype=torch.long, device=args.torch_device),
            "cls": cls.to(args.torch_device),
            "bboxes": boxes.to(args.torch_device),
        }
        with torch.inference_mode():
            _, preds = net(tensor)
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            pred_residual = preds.get("dfl_residual")
            pred_residual = pred_residual.permute(0, 2, 1).contiguous() if pred_residual is not None else None
            anchor_points, stride_tensor = make_anchors(preds["feats"], loss.stride, 0.5)
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=args.torch_device, dtype=pred_scores.dtype) * loss.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = loss.preprocess(targets, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            pred_bboxes = loss.bbox_decode(anchor_points, pred_distri, pred_residual, stride_tensor)
            pred_pixel = pred_bboxes.detach() * stride_tensor
            _, _, target_scores, fg_mask, target_gt_idx = loss.assigner(
                pred_scores.detach().sigmoid(), pred_pixel.type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
            )
        det_boxes, det_conf, det_cls = actual_matches(wrapper, image_path, args)
        n_p2 = math.prod(preds["feats"][0].shape[2:])
        for gt_idx, (area, gt_cls) in enumerate(zip(areas, cls.view(-1).tolist())):
            group = fg_mask[0, :n_p2].bool() & (target_gt_idx[0, :n_p2] == gt_idx)
            q = target_scores[0, :n_p2][group].sum(-1).float()
            gt = gt_bboxes[0, gt_idx]
            candidate_boxes = pred_pixel[0, :n_p2][group]
            if len(q):
                ious = box_iou(candidate_boxes, gt.unsqueeze(0)).view(-1)
                nwd = nwd_similarity(candidate_boxes, gt, args.nwd_c)
                cls_prob = pred_scores[0, :n_p2][group, int(gt_cls)].sigmoid()
                current_alignment = cls_prob.pow(args.tal_alpha) * ious.clamp_min(0).pow(args.tal_beta)
                nwd_alignment = cls_prob.pow(args.tal_alpha) * nwd.pow(args.tal_beta)
                ciou = bbox_iou(candidate_boxes, gt.unsqueeze(0).expand_as(candidate_boxes), xywh=False, CIoU=True).view(-1).clamp(-1, 1)
                raw_box = 1.0 - ciou
                candidate_anchors = (anchor_points * stride_tensor)[:n_p2][group]
                dfl_target = bbox2dist(candidate_anchors, gt.unsqueeze(0).expand_as(candidate_boxes), loss.bbox_loss.dfl_loss.reg_max - 1)
                raw_dfl = loss.bbox_loss.dfl_loss(
                    pred_distri[0, :n_p2][group].view(-1, loss.bbox_loss.dfl_loss.reg_max), dfl_target
                ).mean(-1)
                topq = int(q.argmax())
                oracle = int(ious.argmax())
                top_nwd = int(nwd_alignment.argmax())
                mass = q.sum().clamp_min(1e-12)
                weighted_box = float((q * raw_box).sum() / mass)
                weighted_dfl = float((q * raw_dfl).sum() / mass)
                oracle_iou = float(ious[oracle])
                topq_iou = float(ious[topq])
                mean_iou = float(ious.mean())
                qmax = float(q.max())
                q_mass = float(mass)
                q_mean = float(q.mean())
                qmax_norm_mass = float((q / qmax).sum().item())
                raw_box_mean = float(raw_box.mean())
                raw_dfl_mean = float(raw_dfl.mean())
                top_nwd_iou = float(ious[top_nwd])
                oracle_nwd = float(nwd.max())
                top_current_iou = float(ious[int(current_alignment.argmax())])
                mean_nwd = float(nwd.mean())
            else:
                oracle_iou = topq_iou = mean_iou = qmax = q_mass = q_mean = qmax_norm_mass = 0.0
                weighted_box = weighted_dfl = raw_box_mean = raw_dfl_mean = 0.0
                top_nwd_iou = oracle_nwd = top_current_iou = mean_nwd = 0.0
            keep = det_cls == int(gt_cls)
            if keep.any():
                matched_ious = box_iou(det_boxes[keep], gt_bboxes[0, gt_idx].detach().float().cpu().view(1, 4)).view(-1)
                matched_best = int(matched_ious.argmax())
                actual_iou = float(matched_ious[matched_best])
                actual_conf = float(det_conf[keep][matched_best])
            else:
                actual_iou = actual_conf = 0.0
            failure = "tp75" if actual_iou >= 0.75 else "tp50_only" if actual_iou >= 0.5 else "miss"
            rows.append({
                "image": image_path.name, "gt_index": gt_idx, "class": int(gt_cls),
                "area_px2": area, "sqrt_wh_px": math.sqrt(max(area, 0.0)), "size_group": size_group(area),
                "support_count": int(len(q)), "target_mass": q_mass, "q_mean": q_mean, "q_max": qmax,
                "qmax_normalized_mass": qmax_norm_mass, "n_eff": float(q.square().sum().reciprocal() * q_mass**2) if len(q) else 0.0,
                "oracle_iou": oracle_iou, "topq_iou": topq_iou, "mean_support_iou": mean_iou,
                "oracle_nwd": oracle_nwd, "top_nwd_iou": top_nwd_iou, "top_current_iou": top_current_iou,
                "mean_support_nwd": mean_nwd, "nwd_iou_gap": mean_nwd - mean_iou,
                "oracle_error": 1.0 - oracle_iou, "topq_regret": oracle_iou - topq_iou,
                "raw_box_error_mean": raw_box_mean, "q_weighted_box_error": weighted_box,
                "raw_dfl_error_mean": raw_dfl_mean, "q_weighted_dfl_error": weighted_dfl,
                "actual_best_iou": actual_iou, "actual_best_conf": actual_conf, "failure_group": failure,
            })
        if image_index % 50 == 0:
            print(f"{image_index}/{len(images)} images, {len(rows)} GT", flush=True)
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def aggregate(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    selected = [row for row in rows if key == "all" or row["size_group"] == key]
    out: dict[str, object] = {"gt": len(selected)}
    metrics = ["support_count", "target_mass", "q_mean", "q_max", "qmax_normalized_mass", "n_eff", "oracle_iou", "topq_iou", "mean_support_iou", "oracle_nwd", "top_nwd_iou", "top_current_iou", "mean_support_nwd", "nwd_iou_gap", "oracle_error", "topq_regret", "raw_box_error_mean", "q_weighted_box_error", "raw_dfl_error_mean", "q_weighted_dfl_error", "actual_best_iou", "actual_best_conf"]
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
        out[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        out[f"{metric}_median"] = float(np.median(values)) if len(values) else None
    failures: dict[str, int] = {}
    for row in selected:
        failures[str(row["failure_group"])] = failures.get(str(row["failure_group"]), 0) + 1
    out["failure_counts"] = failures
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-seed", type=int, default=43)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nwd-c", type=float, default=12.8)
    parser.add_argument("--tal-alpha", type=float, default=0.5)
    parser.add_argument("--tal-beta", type=float, default=6.0)
    args = parser.parse_args()
    args.torch_device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = inspect_checkpoint(args.checkpoint, images, args)
    fields = list(rows[0]) if rows else []
    with (args.output / "ftal_diagnostic_per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {group: aggregate(rows, group) for group in ("all", "tiny", "small", "medium", "large")}
    correlations = {}
    for group in ("all", "tiny", "small", "medium", "large"):
        selected = [row for row in rows if group == "all" or row["size_group"] == group]
        correlations[group] = {
            "target_mass_vs_oracle_iou": finite_corr([float(r["target_mass"]) for r in selected], [float(r["oracle_iou"]) for r in selected]),
            "target_mass_vs_oracle_error": finite_corr([float(r["target_mass"]) for r in selected], [float(r["oracle_error"]) for r in selected]),
            "target_mass_vs_actual_iou": finite_corr([float(r["target_mass"]) for r in selected], [float(r["actual_best_iou"]) for r in selected]),
            "target_mass_vs_raw_box_error": finite_corr([float(r["target_mass"]) for r in selected], [float(r["raw_box_error_mean"]) for r in selected]),
        }
    result = {"checkpoint": str(args.checkpoint), "images": len(images), "gt": len(rows), "summary": summary, "correlations": correlations}
    (args.output / "ftal_diagnostic_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
