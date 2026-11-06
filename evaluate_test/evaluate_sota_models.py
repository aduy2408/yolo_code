#!/usr/bin/env python3
"""Evaluation-only TinyPerson runner for the four SOTA model families.

This file never trains, downloads, or guesses a checkpoint/config. Run it once
per explicit model/seed. YOLO models produce merged source-image detections;
MMDetection models consume either an already-merged COCO JSON or corner-window
COCO predictions and delegate the latter to the pinned MMDetection evaluator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("stw", "tph", "mmdet"), required=True)
    p.add_argument("--checkpoint", type=Path, help="Explicit best.pt; required for stw/tph")
    p.add_argument("--predictions", type=Path, help="Explicit COCO predictions JSON; required for mmdet")
    p.add_argument("--predictions-format", choices=("merged-coco", "corner-coco"), default="corner-coco")
    p.add_argument("--test-dir", type=Path, required=True, help="Prepared TinyPerson test corner directory")
    p.add_argument("--data-root", type=Path, required=True, help="TinyPerson root containing annotations/")
    p.add_argument("--native-predictions", type=Path, help="Existing TPH *_predictions.json; skips val.py inference")
    p.add_argument("--tph-python", default=sys.executable, help="Python used only for TPH val.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tph-root", type=Path, help="TPH-YOLOv5 checkout; required for tph")
    p.add_argument("--mmdet-evaluator", type=Path, help="evaluate_tinyperson_metrics.py; required for corner-coco")
    p.add_argument("--device", default="0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nms-iou", type=float, default=0.5)
    return p


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    corner_gt = args.data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    merged_gt = args.data_root / "annotations/task/tiny_set_test_all.json"
    manifest = args.test_dir / "corner_manifest.json"
    for path in (corner_gt, merged_gt, manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    return corner_gt, merged_gt, manifest


def evaluate_mmdet(args: argparse.Namespace, corner_gt: Path, merged_gt: Path) -> dict:
    if args.predictions is None or not args.predictions.is_file():
        raise FileNotFoundError("--predictions is required and must exist for --model mmdet")
    if args.predictions_format == "corner-coco":
        if args.mmdet_evaluator is None or not args.mmdet_evaluator.is_file():
            raise FileNotFoundError("--mmdet-evaluator is required for corner-coco predictions")
        command = [
            sys.executable,
            str(args.mmdet_evaluator),
            "--res", str(args.predictions),
            "--corner-gt", str(corner_gt),
            "--merged-gt", str(merged_gt),
            "--out", str(args.output),
        ]
        completed = subprocess.run(command, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"merged MMDetection evaluator failed with exit={completed.returncode}")
        return json.loads(args.output.read_text())

    from train_scripts.train_all_tinyperson import evaluate_tiny_benchmark

    return evaluate_tiny_benchmark(args.predictions, merged_gt)


def evaluate_stw(args: argparse.Namespace, merged_gt: Path) -> dict:
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("--checkpoint is required and must exist for --model stw")
    from train_scripts.train_all_tinyperson import evaluate_tiny_benchmark, predict_merged_test

    run_dir = args.output.parent / "stw_run"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    link = run_dir / "weights/best.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(args.checkpoint.resolve())
    inference_args = argparse.Namespace(
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
    )
    prediction_path = predict_merged_test(run_dir, args.test_dir, args.data_root, inference_args)
    return evaluate_tiny_benchmark(prediction_path, merged_gt)

def fast_nms(items: list[dict], iou_threshold: float, max_candidates: int = 1000) -> list[dict]:
    """Vectorized class-agnostic NMS, bounded by the COCO maxDets contract."""
    import numpy as np

    ranked = sorted(items, key=lambda item: item["score"], reverse=True)[:max_candidates]
    boxes = np.asarray([item["xyxy"] for item in ranked], dtype=np.float32)
    scores = np.asarray([item["score"] for item in ranked], dtype=np.float32)
    order = scores.argsort()[::-1]
    kept: list[dict] = []
    while order.size:
        current = int(order[0])
        kept.append(ranked[current])
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
        area_current = max(0.0, float(boxes[current, 2] - boxes[current, 0])) * max(0.0, float(boxes[current, 3] - boxes[current, 1]))
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-12)
        order = rest[iou <= iou_threshold]
    return kept


def evaluate_tph(args: argparse.Namespace, manifest_path: Path, merged_gt: Path) -> dict:
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("--checkpoint is required and must exist for --model tph")
    if args.tph_root is None or not args.tph_root.is_dir():
        raise FileNotFoundError("--tph-root is required and must exist for --model tph")
    from train_scripts.train_all_tinyperson import evaluate_tiny_benchmark

    yaml_path = args.test_dir.parent / "tinyperson_split_42_corner_sw640_sh512/tinyperson.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    native_dir = args.output.parent / "tph_native"
    if args.native_predictions is not None:
        prediction_files = [args.native_predictions]
    else:
        compat = Path(__file__).with_name("run_tph_compat.py")
        command = [
            args.tph_python, str(compat), "val.py", "--data", str(yaml_path), "--weights", str(args.checkpoint),
            "--img", str(args.imgsz), "--batch-size", str(args.batch_size), "--device", args.device,
            "--task", "test", "--iou-thres", str(args.nms_iou), "--project", str(native_dir),
            "--name", "test", "--exist-ok", "--verbose", "--save-json",
        ]
        completed = subprocess.run(command, cwd=args.tph_root, text=True, capture_output=True, check=False)
        (args.output.parent / "tph_stdout.log").write_text(completed.stdout)
        (args.output.parent / "tph_stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        prediction_files = sorted(native_dir.rglob("*_predictions.json"))
    if not prediction_files:
        raise RuntimeError("TPH produced no prediction JSON")

    manifest = json.loads(manifest_path.read_text())
    records = {str(item["image_id"]): item for item in manifest["images"]}
    records.update({f"img_{item['image_id']}": item for item in manifest["images"]})
    records.update({f"test_{item['image_id']}": item for item in manifest["images"]})
    merged = json.loads(merged_gt.read_text())
    original_id = {item["file_name"]: item["id"] for item in merged["images"]}
    by_image: dict[int, list[dict]] = {}
    for prediction in json.loads(prediction_files[-1].read_text()):
        record = records.get(str(prediction["image_id"]))
        if record is None:
            continue
        x, y, width, height = prediction["bbox"]
        ox, oy = record["corner"][:2]
        target = original_id[record["file_name"]]
        by_image.setdefault(target, []).append({
            "xyxy": [x + ox, y + oy, x + width + ox, y + height + oy],
            "score": float(prediction["score"]),
        })

    detections = []
    for image_id, items in by_image.items():
        for item in fast_nms(items, args.nms_iou):
            x1, y1, x2, y2 = item["xyxy"]
            detections.append({"image_id": image_id, "category_id": 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1],
                               "score": item["score"]})
    prediction_path = args.output.parent / "tph_test_merged_predictions.json"
    prediction_path.write_text(json.dumps(detections, indent=2) + "\n")
    return evaluate_tiny_benchmark(prediction_path, merged_gt)


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corner_gt, merged_gt, manifest = paths(args)
    if args.model == "stw":
        metrics = evaluate_stw(args, merged_gt)
    elif args.model == "tph":
        metrics = evaluate_tph(args, manifest, merged_gt)
    else:
        metrics = evaluate_mmdet(args, corner_gt, merged_gt)
    record = {
        "model": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "predictions": str(args.predictions) if args.predictions else None,
        "predictions_format": args.predictions_format if args.model == "mmdet" else None,
        "split_seed": 42,
        "training_seed": None,
        "imgsz": args.imgsz,
        "nms_iou": args.nms_iou,
        "protocol": "TinyPerson merged source-image evaluation",
        **metrics,
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
