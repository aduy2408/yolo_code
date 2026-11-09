#!/usr/bin/env python3
"""Evaluate .pt/.onnx/.engine artifacts through Ultralytics predict + COCOeval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from checkpoint_compat import patch_legacy_modules
from ultralytics import YOLO

DATASET_DIRS = {"levirship": "levir_ship", "varroa": "varroa"}
IMAGE_SIZES = {"levirship": 512, "varroa": 640}
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model through Ultralytics predict and COCOeval.")
    parser.add_argument("--model", type=Path, required=True, help=".pt, .onnx, or Jetson-native .engine")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRS), required=True)
    parser.add_argument("--split", choices=("val", "test"), nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_path(data_root: Path, dataset: str, split: str) -> Path:
    return data_root / DATASET_DIRS[dataset] / "annotations" / "seed_42" / f"{split}.json"


def predict(model: YOLO, payload: dict, data_root: Path, dataset: str, args: argparse.Namespace) -> list[dict]:
    image_root = data_root / DATASET_DIRS[dataset]
    detections: list[dict] = []
    for index, image in enumerate(payload["images"], start=1):
        path = image_root / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        result = model.predict(
            source=str(path),
            imgsz=IMAGE_SIZES[dataset],
            batch=1,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=args.precision == "fp16",
            verbose=False,
            save=False,
        )[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            for box, score, category in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(float, box)
                detections.append({
                    "image_id": int(image["id"]),
                    "category_id": int(category) + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                })
        if index % 100 == 0:
            print(f"Predicted {index}/{len(payload['images'])} images", flush=True)
    return detections


def coco_metrics(annotation: Path, detections: list[dict]) -> dict[str, float]:
    coco_gt = COCO(str(annotation))
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
    metrics: dict[str, float] = {}
    for label, area, suffix in (
        ("all", [0.0, 1e12], "all"),
        ("small", [20.0**2, 32.0**2], "Small"),
    ):
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.iouThrs = IOU_THRESHOLDS
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.params.areaRng = [area]
        evaluator.params.areaRngLbl = [label]
        evaluator.evaluate()
        precision = evaluator.eval["precision"]
        ap50_values = precision[0, :, :, 0, -1]
        ap_values = precision[:, :, :, 0, -1]
        metrics[f"AP50-{suffix}"] = float(ap50_values[ap50_values > -1].mean())
        metrics[f"mAP50-95-{suffix}"] = float(ap_values[ap_values > -1].mean())
    return metrics


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model = YOLO(str(args.model), task="detect")
    patched_modules = patch_legacy_modules(model)
    if patched_modules:
        print(f"Applied inference compatibility defaults: {patched_modules}", flush=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    for split in args.split:
        annotation = annotation_path(args.data_root, args.dataset, split)
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        detections = predict(model, payload, args.data_root, args.dataset, args)
        stem = f"{args.model.stem}_{split}"
        prediction_path = args.results_root / f"{stem}_predictions.json"
        metrics_path = args.results_root / f"{stem}_metrics.json"
        prediction_path.write_text(json.dumps(detections) + "\n", encoding="utf-8")
        row = {
            "model": str(args.model),
            "model_sha256": sha256(args.model),
            "dataset": args.dataset,
            "split": split,
            "provider": "Ultralytics YOLO.predict",
            "precision": args.precision,
            "image_size": IMAGE_SIZES[args.dataset],
            "batch_size": 1,
            "conf": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
            "prediction_file": str(prediction_path),
            "evaluator": "pycocotools.COCOeval",
            "protocol": "Ultralytics predict preprocessing/decode/NMS followed by COCOeval",
            **coco_metrics(annotation, detections),
        }
        metrics_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
