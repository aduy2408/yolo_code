#!/usr/bin/env python3
"""Calibrate fixed train-split quantiles for context-adaptive OACP policies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_ultralytics.context_augment import (
    _boxes,
    _contrast_adaptive_expands,
    _far_context_richness,
    _mask_from_boxes,
)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("calibration produced no finite values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def calibrate(dataset_root: Path, output: Path, split: str = "train") -> dict[str, object]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"missing prepared split: {image_dir}")
    richness: list[float] = []
    contrast: list[float] = []
    image_count = 0
    object_count = 0
    for image_path in sorted(image_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        label_path = label_dir / f"{image_path.stem}.txt"
        if image is None or not label_path.is_file():
            continue
        h, w = image.shape[:2]
        rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) == 5:
                _, xc, yc, bw, bh = map(float, values)
                rows.append([xc, yc, bw, bh])
        boxes = _boxes({"bboxes": np.asarray(rows, dtype=np.float32)}, h, w)
        sizes = np.sqrt(np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.empty(0)
        tiny = boxes[sizes < 32.0]
        protected = _mask_from_boxes(tiny, h, w, 3.0).astype(bool)
        protected |= _mask_from_boxes(boxes, h, w, 1.2).astype(bool)
        value = _far_context_richness(image, protected)
        if np.isfinite(value):
            richness.append(float(value))
        if len(tiny):
            _, records = _contrast_adaptive_expands(image, boxes, np.flatnonzero(sizes < 32.0), measurement_expand=2.5)
            contrast.extend(float(record["contrast_raw"]) for record in records if record["contrast_raw"] is not None and np.isfinite(record["contrast_raw"]))
            object_count += len(records)
        image_count += 1
    result = {
        "split": split,
        "image_count": image_count,
        "eligible_object_count": object_count,
        "context_richness_q10": _quantiles(richness)["q10"],
        "context_richness_q50": _quantiles(richness)["q50"],
        "context_richness_q90": _quantiles(richness)["q90"],
        "local_contrast_q10": _quantiles(contrast)["q10"],
        "local_contrast_q50": _quantiles(contrast)["q50"],
        "local_contrast_q90": _quantiles(contrast)["q90"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    print(json.dumps(calibrate(args.dataset_root, args.output, args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
