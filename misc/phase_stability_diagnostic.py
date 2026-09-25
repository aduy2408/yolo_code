#!/usr/bin/env python3
"""Measure tiny-object prediction stability under sub-stride image translations.

The protocol keeps the pixels and detector fixed, translates the whole image by
``dx,dy in [0, phase_stride)`` with zero padding, and translates predictions
back to the original coordinate system.  Each ground-truth object is therefore
observed across the same 4x4 phase orbit.  The report separates oracle
localization, score/ranking, and optional intermediate-feature stability.

This is intentionally inference-only.  It does not retrain or modify a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def local_ultralytics() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    package = ROOT / "models_related/ultralytics"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return pairwise xyxy IoU for two arrays, including empty inputs."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0.0), axis=1)[:, None]
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0.0), axis=1)[None]
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """Read class cx cy w h labels and return absolute xyxy boxes."""
    rows: list[list[float]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 5:
                rows.append([float(value) for value in values[:5]])
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    values = np.asarray(rows, dtype=np.float64)
    cx, cy, bw, bh = values[:, 1:].T * np.asarray([width, height, width, height])[:, None]
    return np.column_stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))


def label_path_for_image(image: Path, labels_root: Path | None = None) -> Path:
    if labels_root is not None:
        return labels_root / image.with_suffix(".txt").name
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def translate_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate into a larger zero-padded canvas without cropping any pixels."""
    if dx < 0 or dy < 0:
        raise ValueError("This protocol expects non-negative phase offsets")
    height, width = image.shape[:2]
    output = np.zeros((height + dy, width + dx, *image.shape[2:]), dtype=image.dtype)
    output[dy:, dx:] = image
    return output


def translate_boxes(boxes: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.asarray(boxes, dtype=np.float64).copy().reshape(-1, 4)
    if len(shifted):
        shifted[:, [0, 2]] += dx
        shifted[:, [1, 3]] += dy
    return shifted


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def size_bucket(box: np.ndarray) -> str:
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    diagonal = math.hypot(width, height)
    if diagonal < 8:
        return "tiny_lt8"
    if diagonal < 12:
        return "tiny_8_12"
    if diagonal < 20:
        return "tiny_12_20"
    if diagonal < 32:
        return "small_20_32"
    return "large_ge32"


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "std": None, "median": None, "q90": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _feature_vector(output: Any) -> np.ndarray | None:
    """Convert a hooked module output to a stable global vector."""
    import torch

    if isinstance(output, (tuple, list)):
        output = next((item for item in output if torch.is_tensor(item)), None)
    if not torch.is_tensor(output):
        return None
    tensor = output.detach().float()
    if tensor.ndim == 4:
        tensor = tensor.mean(dim=(-2, -1))
    return tensor[0].cpu().numpy().astype(np.float64, copy=False).ravel()


def _module_by_name(model: Any, name: str) -> Any:
    modules = dict(model.model.named_modules())
    if name not in modules:
        examples = ", ".join(list(modules)[:12])
        raise KeyError(f"Unknown feature layer {name!r}; examples: {examples}")
    return modules[name]


def _predict_one(model: Any, image: np.ndarray, *, imgsz: int, conf: float, device: str,
                 feature_layer: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    captured: list[np.ndarray] = []
    handle = None
    if feature_layer:
        handle = _module_by_name(model, feature_layer).register_forward_hook(
            lambda _module, _inputs, output: captured.append(_feature_vector(output))
        )
    try:
        result = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    finally:
        if handle is not None:
            handle.remove()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,)), captured[-1] if captured else None
    return (
        boxes.xyxy.detach().cpu().numpy().astype(np.float64),
        boxes.conf.detach().cpu().numpy().astype(np.float64),
        captured[-1] if captured else None,
    )


def _object_rows(gt: np.ndarray, predictions: list[dict[str, Any]], phases: list[tuple[int, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_index, ground_truth in enumerate(gt):
        phase_values = []
        for prediction, (dx, dy) in zip(predictions, phases):
            boxes = prediction["boxes"] - np.asarray([dx, dy, dx, dy])
            scores = prediction["scores"]
            ious = box_iou(ground_truth[None], boxes)[0]
            oracle = float(ious.max()) if len(ious) else 0.0
            best_iou_index = int(np.argmax(ious)) if len(ious) else -1
            top_score_index = int(np.argmax(scores)) if len(scores) else -1
            phase_values.append({
                "dx": dx,
                "dy": dy,
                "oracle_iou": oracle,
                "best_iou_score": float(scores[best_iou_index]) if best_iou_index >= 0 else 0.0,
                "top_score": float(scores[top_score_index]) if top_score_index >= 0 else 0.0,
                "top_score_iou": float(ious[top_score_index]) if top_score_index >= 0 else 0.0,
            })
        oracle_values = [value["oracle_iou"] for value in phase_values]
        score_values = [value["top_score"] for value in phase_values]
        top_iou_values = [value["top_score_iou"] for value in phase_values]
        best_score_values = [value["best_iou_score"] for value in phase_values]
        rows.append({
            "object_index": object_index,
            "width": float(ground_truth[2] - ground_truth[0]),
            "height": float(ground_truth[3] - ground_truth[1]),
            "diagonal": float(math.hypot(ground_truth[2] - ground_truth[0], ground_truth[3] - ground_truth[1])),
            "size_bucket": size_bucket(ground_truth),
            "oracle_iou_std": float(np.std(oracle_values)),
            "oracle_iou_min": float(np.min(oracle_values)),
            "oracle_iou_max": float(np.max(oracle_values)),
            "top_score_std": float(np.std(score_values)),
            "top_score_iou_std": float(np.std(top_iou_values)),
            "best_iou_score_std": float(np.std(best_score_values)),
            "phases": phase_values,
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    local_ultralytics()
    from PIL import Image
    from ultralytics import YOLO

    images = sorted(path for path in args.images.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")
    phases = [(dx, dy) for dy in range(args.phase_stride) for dx in range(args.phase_stride)]
    wrapper = YOLO(str(args.weights))
    model = wrapper.model
    rows: list[dict[str, Any]] = []
    feature_cosines: list[float] = []
    for image_path in images:
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        height, width = image.shape[:2]
        gt = read_yolo_labels(label_path_for_image(image_path, args.labels), width, height)
        predictions: list[dict[str, Any]] = []
        for dx, dy in phases:
            shifted = translate_image(image, dx, dy)
            boxes, scores, feature = _predict_one(
                wrapper, shifted, imgsz=args.imgsz, conf=args.conf, device=args.device,
                feature_layer=args.feature_layer,
            )
            predictions.append({"boxes": boxes, "scores": scores, "feature": feature})
        object_rows = _object_rows(gt, predictions, phases)
        if args.feature_layer and predictions and predictions[0]["feature"] is not None:
            reference = predictions[0]["feature"]
            feature_cosines.extend(
                cosine_similarity(reference, item["feature"])
                for item in predictions[1:] if item["feature"] is not None
            )
        rows.extend({"image": image_path.name, **row} for row in object_rows)
        if len(rows) and len(rows) % 100 == 0:
            print(f"processed {image_path.name}: {len(images)} images")

    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in sorted({row["size_bucket"] for row in rows}):
        bucket_rows = [row for row in rows if row["size_bucket"] == bucket]
        by_bucket[bucket] = {
            "objects": len(bucket_rows),
            "oracle_iou_std": summarize(row["oracle_iou_std"] for row in bucket_rows),
            "oracle_iou_min": summarize(row["oracle_iou_min"] for row in bucket_rows),
            "top_score_std": summarize(row["top_score_std"] for row in bucket_rows),
            "top_score_iou_std": summarize(row["top_score_iou_std"] for row in bucket_rows),
            "best_iou_score_std": summarize(row["best_iou_score_std"] for row in bucket_rows),
        }
    report: dict[str, Any] = {
        "protocol": {
            "name": "translation_phase_sweep",
            "phase_offsets": phases,
            "phase_stride": args.phase_stride,
            "padding": "zero; expanded canvas, no crop",
            "prediction_coordinates": "translated back by subtracting dx,dy",
            "oracle_iou": "maximum prediction IoU with each original-coordinate GT box",
            "top_score_iou": "IoU of highest-confidence prediction after coordinate compensation",
        },
        "weights": str(args.weights),
        "images": len(images),
        "objects": len(rows),
        "metrics": {
            "oracle_iou_std": summarize(row["oracle_iou_std"] for row in rows),
            "top_score_std": summarize(row["top_score_std"] for row in rows),
            "top_score_iou_std": summarize(row["top_score_iou_std"] for row in rows),
            "best_iou_score_std": summarize(row["best_iou_score_std"] for row in rows),
        },
        "by_size_bucket": by_bucket,
        "feature": {
            "layer": args.feature_layer,
            "global_cosine_to_phase_0": summarize(feature_cosines) if args.feature_layer else None,
        },
        "per_object": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "object_index", "width", "height", "diagonal", "size_bucket",
                  "oracle_iou_std", "oracle_iou_min", "oracle_iou_max", "top_score_std",
                  "top_score_iou_std", "best_iou_score_std"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Optional flat YOLO label directory for source images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase-stride", type=int, default=4)
    parser.add_argument("--feature-layer", help="Optional named model module for global feature cosine diagnostics")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase_stride < 1:
        parser.error("--phase-stride must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args())["metrics"], indent=2))
