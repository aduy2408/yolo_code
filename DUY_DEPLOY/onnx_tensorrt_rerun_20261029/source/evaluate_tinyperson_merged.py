from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TensorRT YOLO outputs with TinyPerson corner merge.")
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--trt-cache-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-only", action="store_true", help="Save crop predictions without coordinate merge or metric evaluation.")
    parser.add_argument("--model-path", type=Path, help="Evaluate one ONNX model instead of the complete model tree.")
    return parser.parse_args()


def nms(items: list[tuple[np.ndarray, float]], threshold: float, max_det: int) -> list[tuple[np.ndarray, float]]:
    if not items:
        return []
    boxes = np.asarray([item[0] for item in items], dtype=np.float32)
    scores = np.asarray([item[1] for item in items], dtype=np.float32)
    order = np.argsort(-scores)
    kept: list[int] = []
    while order.size and len(kept) < max_det:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        candidates = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[candidates, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[candidates, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[candidates, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[candidates, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(0.0, boxes[current, 3] - boxes[current, 1])
        area_candidates = np.maximum(0.0, boxes[candidates, 2] - boxes[candidates, 0]) * np.maximum(0.0, boxes[candidates, 3] - boxes[candidates, 1])
        overlap = intersection / np.maximum(area_current + area_candidates - intersection, 1e-12)
        order = candidates[overlap <= threshold]
    return [(boxes[index], float(scores[index])) for index in kept]


def providers(cache_root: Path) -> list[object]:
    name = "TensorrtExecutionProvider"
    if name not in ort.get_available_providers():
        raise RuntimeError(f"{name} unavailable: {ort.get_available_providers()}")
    cache_root.mkdir(parents=True, exist_ok=True)
    return [(name, {"device_id": 0, "trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_root)})]


def main() -> None:
    args = parse_args()
    data_root = args.data_root / "tinyperson"
    corner_path = data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json"
    merged_path = data_root / "annotations/task/tiny_set_test_all.json"
    corner = json.loads(corner_path.read_text())
    merged = json.loads(merged_path.read_text())
    original_id = {row["file_name"]: int(row["id"]) for row in corner["old_images"]}
    records = sorted(corner["images"], key=lambda row: int(row["id"]))
    crop_images = [{"id": int(row["id"]), "width": int(row["width"]), "height": int(row["height"]), "path": data_root / "images/all" / f"test_{int(row['id']):06d}.jpg"} for row in records]
    ground_truth: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in merged["annotations"]:
        x, y, width, height = row["bbox"]
        ground_truth[int(row["image_id"])].append(np.asarray([x, y, x + width, y + height], dtype=np.float32))
    images = [{**row, "path": data_root / "images/all" / f"test_{int(row['id']):06d}.jpg"} for row in merged["images"]]
    provider_settings = providers(args.trt_cache_root)
    args.results_root.mkdir(parents=True, exist_ok=True)
    model_paths = [args.model_path] if args.model_path else sorted(args.models_root.rglob("*.onnx"))
    for model_path in model_paths:
        options = ort.SessionOptions()
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=provider_settings)
        if hasattr(session, "disable_fallback"):
            session.disable_fallback()
        if session.get_providers()[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"TensorRT was not first for {model_path}: {session.get_providers()}")
        crop_predictions = evaluate.predictions(session, crop_images, 640, args.batch_size, args.conf, args.nms_iou, args.max_det)
        relative = model_path.relative_to(args.models_root)
        prediction_output = args.predictions_root / relative.with_suffix(".npz")
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        crop_ids, boxes, scores = [], [], []
        for crop_id, detections in crop_predictions.items():
            for box, score in detections:
                crop_ids.append(int(crop_id))
                boxes.append(np.asarray(box, dtype=np.float32))
                scores.append(float(score))
        np.savez_compressed(
            prediction_output,
            crop_ids=np.asarray(crop_ids, dtype=np.int32),
            boxes=np.asarray(boxes, dtype=np.float32).reshape((-1, 4)),
            scores=np.asarray(scores, dtype=np.float32),
        )
        prediction_metadata = prediction_output.with_suffix(".json")
        prediction_metadata.write_text(json.dumps({
            "model": str(model_path),
            "prediction_file": str(prediction_output),
            "dataset": "tinyperson",
            "split": "test_corner_windows",
            "save_only": args.save_only,
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "batch_size": args.batch_size,
            "provider": "TensorrtExecutionProvider",
        }, indent=2, sort_keys=True) + "\n")
        if args.save_only:
            continue
        by_original: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for row in records:
            original = original_id[row["file_name"]]
            dx, dy = row["corner"][:2]
            for box, score in crop_predictions.get(int(row["id"]), []):
                by_original[original].append((box + np.asarray([dx, dy, dx, dy], dtype=np.float32), score))
        merged_predictions = {key: nms(value, args.nms_iou, args.max_det) for key, value in by_original.items()}
        metrics = evaluate.evaluate(images, ground_truth, merged_predictions)
        row = {"model": str(model_path), "dataset": "tinyperson", "split": "test_merged", "provider": "TensorrtExecutionProvider", "batch_size": args.batch_size, "nms_iou": args.nms_iou, "protocol": "corner-window inference, original-coordinate translation, one-class NMS merge", **metrics}
        relative = model_path.relative_to(args.models_root)
        output = args.results_root / "tinyperson" / "test_merged" / relative.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
