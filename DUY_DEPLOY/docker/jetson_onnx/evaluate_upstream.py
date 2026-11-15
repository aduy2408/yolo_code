#!/usr/bin/env python3
"""Evaluate saved Jetson predictions with the upstream COCO protocol.

Inference remains in evaluate.py. This script only converts the existing .npz
artifacts to COCO detections and runs pycocotools, so old custom-score results
remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET_DIRS = {"levirship": "levir_ship", "varroa": "varroa"}
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved ONNX predictions with COCOeval.")
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRS), required=True)
    parser.add_argument("--split", choices=("val", "test"), nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def annotation_path(data_root: Path, dataset: str, split: str) -> Path:
    return data_root / DATASET_DIRS[dataset] / "annotations" / "seed_42" / f"{split}.json"


def detections_from_npz(path: Path) -> list[dict]:
    raw = np.load(path)
    image_ids = raw["image_ids"]
    boxes = raw["boxes"]
    scores = raw["scores"]
    if not (len(image_ids) == len(boxes) == len(scores)):
        raise ValueError(f"Prediction arrays have inconsistent lengths: {path}")
    detections = []
    for image_id, box, score in zip(image_ids, boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            }
        )
    return detections


def evaluate_one(annotation: Path, detections: list[dict]) -> dict[str, float]:
    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    metrics: dict[str, float] = {}
    for label, area, suffix in (
        ("all", [0.0, 1e12], "all"),
        ("small", [0.0, 32.0**2], "Small"),
    ):
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = IOU_THRESHOLDS
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.params.areaRng = [area]
        evaluator.params.areaRngLbl = [label]
        evaluator.evaluate()
        evaluator.accumulate()
        precision = evaluator.eval["precision"][:, :, :, 0, -1]
        valid = precision > -1
        metrics[f"AP50-{suffix}"] = float(precision[0][precision[0] > -1].mean())
        metrics[f"mAP50-95-{suffix}"] = float(precision[valid].mean())
    return metrics


def main() -> None:
    args = parse_args()
    prediction_root = args.predictions_root / args.dataset
    for split in args.split:
        annotation = annotation_path(args.data_root, args.dataset, split)
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        split_root = prediction_root / split
        paths = sorted(split_root.rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz predictions under {split_root}")
        for prediction_path in paths:
            metrics = evaluate_one(annotation, detections_from_npz(prediction_path))
            relative = prediction_path.relative_to(prediction_root).with_suffix(".json")
            output = args.results_root / args.dataset / split / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "model": str(prediction_path),
                "prediction_file": str(prediction_path),
                "annotation_file": str(annotation),
                "dataset": args.dataset,
                "split": split,
                "evaluator": "pycocotools.COCOeval",
                "nms_iou": args.nms_iou,
                "protocol": "COCOeval on saved TensorRT predictions; inference/NMS already completed by evaluate.py",
                **metrics,
            }
            output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
