#!/usr/bin/env python3
"""Test whether raw localization instability is periodic with a stride-4 lattice.

For each image, this inference-only probe translates the image along x by
``dx=0..7`` (expanded zero-padded canvas, ``dy=0``), runs the detector without
NMS, translates every decoded box back to the original coordinate system, and
computes the oracle IoU for each valid ground-truth object.  It reports:

``D1 = mean(abs(I(dx+1)-I(dx)))`` for dx=0..6
``D4 = mean(abs(I(dx+4)-I(dx)))`` for dx=0..3

The decision rule is ``D4 < 0.5 * D1`` on the unstable top quartile.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from misc.phase_stability_diagnostic import (  # noqa: E402
    box_iou,
    label_path_for_image,
    read_yolo_labels,
    size_bucket,
    summarize,
    translate_image,
)


def _local_ultralytics() -> None:
    package = ROOT / "models_related/ultralytics"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))


def periodicity_distances(oracle_ious: Iterable[float]) -> tuple[float, float]:
    values = np.asarray(list(oracle_ious), dtype=np.float64)
    if values.shape != (8,):
        raise ValueError(f"Expected exactly 8 oracle IoUs, got shape {values.shape}")
    d1 = float(np.abs(np.diff(values)).mean())
    d4 = float(np.abs(values[4:] - values[:4]).mean())
    return d1, d4


def valid_for_x_sweep(box: np.ndarray, width: int, height: int, max_dx: int = 7) -> bool:
    """Keep objects away from both x-side padding boundaries for every phase."""
    x1, y1, x2, y2 = map(float, box)
    return x1 >= max_dx and y1 >= 0 and x2 <= width - max_dx and y2 <= height


def _raw_boxes(model: Any, image: np.ndarray, *, imgsz: int, device: str) -> np.ndarray:
    """Run the local YOLO model and return all decoded xyxy boxes before NMS."""
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.ops import scale_boxes, xywh2xyxy

    height, width = image.shape[:2]
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    transformed = letterbox(image=image)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed)).to(device)
    tensor = tensor.permute(2, 0, 1).float()[None] / 255.0
    with torch.inference_mode():
        decoded, _raw = model.model(tensor)
    if isinstance(decoded, (tuple, list)):
        decoded = decoded[0]
    if decoded.ndim != 3 or decoded.shape[1] < 4:
        raise RuntimeError(f"Unexpected decoded output shape: {tuple(decoded.shape)}")
    boxes = xywh2xyxy(decoded[0, :4].T).float()
    ratio = min(imgsz / height, imgsz / width)
    pad = ((imgsz - round(width * ratio)) / 2, (imgsz - round(height * ratio)) / 2)
    boxes = scale_boxes((imgsz, imgsz), boxes, (height, width), ratio_pad=((ratio, ratio), pad))
    return boxes.detach().cpu().numpy().astype(np.float64)


def _object_rows(
    gt: np.ndarray,
    phase_boxes: list[np.ndarray],
    phases: list[int],
    width: int,
    height: int,
    unstable_quantile: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for object_index, box in enumerate(gt):
        if not valid_for_x_sweep(box, width, height):
            continue
        ious: list[float] = []
        for dx, boxes in zip(phases, phase_boxes):
            restored = boxes - np.asarray([dx, 0, dx, 0], dtype=np.float64)
            values = box_iou(box[None], restored)[0]
            ious.append(float(values.max()) if len(values) else 0.0)
        d1, d4 = periodicity_distances(ious)
        candidates.append({
            "object_index": object_index,
            "width": float(box[2] - box[0]),
            "height": float(box[3] - box[1]),
            "diagonal": float(np.hypot(box[2] - box[0], box[3] - box[1])),
            "size_bucket": size_bucket(box),
            "oracle_iou_std": float(np.std(ious)),
            "oracle_iou_min": float(np.min(ious)),
            "oracle_iou_max": float(np.max(ious)),
            "d1": d1,
            "d4": d4,
            "d4_over_d1": d4 / d1 if d1 > 1e-12 else None,
            "oracle_ious": ious,
        })
    return candidates


def _summarize_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    unstable = [row for row in rows if row["oracle_iou_std"] >= threshold]
    selected = unstable if unstable else rows
    ratios = [row["d4_over_d1"] for row in selected if row["d4_over_d1"] is not None]
    return {
        "objects": len(selected),
        "d1": summarize(row["d1"] for row in selected),
        "d4": summarize(row["d4"] for row in selected),
        "d4_over_d1": summarize(ratios),
        "fraction_d4_lt_half_d1": float(np.mean([row["d4"] < 0.5 * row["d1"] for row in selected])) if selected else None,
        "threshold_oracle_iou_std": threshold,
        "selection": "top unstable quartile by oracle_iou_std" if unstable else "all valid objects (fewer than four rows)",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _local_ultralytics()
    from PIL import Image
    from ultralytics import YOLO

    images = sorted(path for path in args.images.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")
    phases = list(range(8))
    model = YOLO(str(args.weights))
    rows: list[dict[str, Any]] = []
    considered = 0
    rejected = 0
    for image_path in images:
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        height, width = image.shape[:2]
        gt = read_yolo_labels(label_path_for_image(image_path, args.labels), width, height)
        considered += len(gt)
        rejected += sum(not valid_for_x_sweep(box, width, height) for box in gt)
        phase_boxes = [_raw_boxes(model, translate_image(image, dx, 0), imgsz=args.imgsz, device=args.device) for dx in phases]
        rows.extend({"image": image_path.name, **row} for row in _object_rows(gt, phase_boxes, phases, width, height, args.unstable_quantile))
    if not rows:
        raise RuntimeError("No border-valid objects survived the x sweep")
    threshold = float(np.quantile([row["oracle_iou_std"] for row in rows], args.unstable_quantile))
    selected = [row for row in rows if row["oracle_iou_std"] >= threshold]
    d1 = [row["d1"] for row in selected]
    d4 = [row["d4"] for row in selected]
    keep = bool(selected) and float(np.mean(np.asarray(d4) < 0.5 * np.asarray(d1))) > args.keep_fraction
    report: dict[str, Any] = {
        "protocol": {
            "name": "stride_periodicity_raw_pre_nms",
            "translations": {"axis": "x", "dx": phases, "dy": 0},
            "stride": 4,
            "padding": "zero; expanded canvas, no crop",
            "boxes": "all decoded boxes before NMS and confidence filtering",
            "coordinate_compensation": "subtract dx from x1 and x2",
            "validity": "GT x1 >= 7 and x2 <= width-7; no x-side padding contact across all phases",
            "D1": "mean absolute adjacent IoU change over dx=0..6",
            "D4": "mean absolute IoU change between dx and dx+4 over dx=0..3",
            "decision_rule": "KEEP iff more than 50% of top-quartile unstable objects satisfy D4 < 0.5*D1",
        },
        "weights": str(args.weights),
        "images": len(images),
        "gt_objects_considered": considered,
        "border_rejected_objects": rejected,
        "valid_objects": len(rows),
        "unstable_selection": _summarize_rows(rows, threshold),
        "decision": {
            "label": "KEEP" if keep else "KILL",
            "top_quartile_objects": len(selected),
            "fraction_d4_lt_half_d1": float(np.mean(np.asarray(d4) < 0.5 * np.asarray(d1))) if selected else None,
            "keep_fraction_threshold": args.keep_fraction,
            "threshold_oracle_iou_std": threshold,
        },
        "per_object": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "object_index", "width", "height", "diagonal", "size_bucket", "oracle_iou_std", "oracle_iou_min", "oracle_iou_max", "d1", "d4", "d4_over_d1", "oracle_ious"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields}
            output["oracle_ious"] = json.dumps(output["oracle_ious"])
            writer.writerow(output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--unstable-quantile", type=float, default=0.75)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.unstable_quantile < 1:
        parser.error("--unstable-quantile must be in [0, 1)")
    if not 0 < args.keep_fraction <= 1:
        parser.error("--keep-fraction must be in (0, 1]")
    return args


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps(report["decision"], indent=2))
