#!/usr/bin/env python3
"""Compare TAL positive ordering reliability between plain P2 and GAP P2."""

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
from ultralytics.utils.loss import make_anchors, v8DetectionLoss  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402


VARIANTS = {"plain_p2_only", "gap_baseline"}
METRICS = ("spearman_r_iou", "top_r_iou", "oracle_iou", "top_r_regret", "top1_agreement")


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def gt_group(area: float) -> str:
    return "tiny" if area < 100 else "small" if area <= 400 else "large"


def read_labels(image: Path) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    rows = [line.split() for line in labels_for(image).read_text().splitlines() if line.strip()]
    cls = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32).reshape(-1, 1)
    boxes = torch.tensor([[float(value) for value in row[1:5]] for row in rows], dtype=torch.float32).reshape(-1, 4)
    original = cv2.imread(str(image))
    if original is None:
        raise RuntimeError(f"could not read image: {image}")
    height, width = original.shape[:2]
    areas = [float(box[2] * width * box[3] * height) for box in boxes]
    return cls, boxes, areas


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 2:
        return math.nan
    ra, rb = average_ranks(a.detach().cpu().numpy()), average_ranks(b.detach().cpu().numpy())
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def reliability_metrics(q: torch.Tensor, iou: torch.Tensor) -> dict[str, float | int]:
    positive = q > 0
    q, iou = q[positive].float(), iou[positive].float()
    if not len(q):
        return {
            "support_count": 0,
            "spearman_r_iou": math.nan,
            "top_r_iou": math.nan,
            "oracle_iou": math.nan,
            "top_r_regret": math.nan,
            "top1_agreement": math.nan,
        }
    r = q / q.max().clamp_min(1e-12)
    top_r = int(r.argmax())
    oracle = int(iou.argmax())
    oracle_iou = float(iou[oracle])
    top_r_iou = float(iou[top_r])
    return {
        "support_count": int(len(q)),
        "spearman_r_iou": spearman(r, iou),
        "top_r_iou": top_r_iou,
        "oracle_iou": oracle_iou,
        "top_r_regret": oracle_iou - top_r_iou,
        "top1_agreement": int(top_r == oracle),
    }


def inspect_checkpoint(name: str, checkpoint: Path, images: list[Path], args: argparse.Namespace) -> list[dict[str, object]]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if args.expected_seed is not None and train_args.get("seed") != args.expected_seed:
        raise RuntimeError(f"{name}: expected checkpoint seed {args.expected_seed}, got {train_args.get('seed')!r}")
    net = wrapper.model.to(args.torch_device).eval()
    head = net.model[-1]
    strides = [float(value) for value in head.stride]
    if not strides or strides[0] != 4.0:
        raise RuntimeError(f"{name}: expected P2 stride first, got {strides}")

    loss = v8DetectionLoss(net)
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    rows = []
    for image_index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"could not read image: {image_path}")
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.torch_device).permute(2, 0, 1).float()[None] / 255
        cls, boxes, areas = read_labels(image_path)
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
                pred_scores.detach().sigmoid(),
                pred_pixel.type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        n_p2 = math.prod(preds["feats"][0].shape[2:])
        for gt_idx, area in enumerate(areas):
            group = fg_mask[0, :n_p2].bool() & (target_gt_idx[0, :n_p2] == gt_idx)
            q = target_scores[0, :n_p2][group].sum(-1)
            iou = box_iou(gt_bboxes[0, gt_idx][None], pred_pixel[0, :n_p2][group])[0] if group.any() else q
            rows.append(
                {
                    "variant": name,
                    "image": image_path.name,
                    "gt_index": gt_idx,
                    "area_px2": area,
                    "size_group": gt_group(area),
                    **reliability_metrics(q, iou),
                }
            )
        if image_index % 100 == 0:
            print(f"{name}: {image_index}/{len(images)} images, {len(rows)} GT", flush=True)
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        out[variant] = {}
        variant_rows = [row for row in rows if row["variant"] == variant]
        for group in ("all", "tiny", "small", "large"):
            selected = variant_rows if group == "all" else [row for row in variant_rows if row["size_group"] == group]
            record: dict[str, float | int | None] = {"gt": len(selected)}
            for metric in ("support_count", *METRICS):
                values = np.asarray([float(row[metric]) for row in selected if math.isfinite(float(row[metric]))], dtype=np.float64)
                record[f"{metric}_mean"] = float(values.mean()) if len(values) else None
                record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            out[variant][group] = record
    return out


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal/tal_r_iou_reliability")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", action="append", required=True, help="variant=/path/to/best.pt")
    args = parser.parse_args()
    args.torch_device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
    checkpoints = parse_checkpoints(args.checkpoint)
    missing = sorted(VARIANTS - set(checkpoints))
    if missing:
        raise ValueError(f"missing checkpoints: {missing}")
    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, checkpoint in checkpoints.items():
        rows.extend(inspect_checkpoint(name, checkpoint, images, args))
    fields = ["variant", "image", "gt_index", "area_px2", "size_group", "support_count", *METRICS]
    with (args.output / "tal_r_iou_reliability_per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "tal_r_iou_reliability_summary.json").write_text(
        json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
