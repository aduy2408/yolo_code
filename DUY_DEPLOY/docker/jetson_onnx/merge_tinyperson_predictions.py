from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge saved TinyPerson corner predictions offline.")
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root / "tinyperson"
    corner = json.loads((data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json").read_text())
    merged = json.loads((data_root / "annotations/task/tiny_set_test_all.json").read_text())
    original_id = {row["file_name"]: int(row["id"]) for row in corner["old_images"]}
    records = {int(row["id"]): row for row in corner["images"]}
    ground_truth: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in merged["annotations"]:
        x, y, width, height = row["bbox"]
        ground_truth[int(row["image_id"])].append(np.asarray([x, y, x + width, y + height], dtype=np.float32))
    images = [{**row, "path": data_root / "images/all" / f"test_{int(row['id']):06d}.jpg"} for row in merged["images"]]

    for prediction_path in sorted(args.predictions_root.rglob("*.npz")):
        relative = prediction_path.relative_to(args.predictions_root)
        model_key = relative.with_suffix("")
        metadata_path = prediction_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        raw = np.load(prediction_path)
        by_original: dict[int, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for crop_id, box, score in zip(raw["crop_ids"], raw["boxes"], raw["scores"]):
            row = records[int(crop_id)]
            original = original_id[row["file_name"]]
            dx, dy = row["corner"][:2]
            by_original[original].append((box + np.asarray([dx, dy, dx, dy], dtype=np.float32), float(score)))
        merged_predictions = {key: nms(value, args.nms_iou, args.max_det) for key, value in by_original.items()}
        metrics = evaluate.evaluate(images, ground_truth, merged_predictions)
        output = args.results_root / "tinyperson" / "test_merged" / model_key.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "model": metadata.get("model", str(model_key)),
            "prediction_file": str(prediction_path),
            "dataset": "tinyperson",
            "split": "test_merged",
            "provider": metadata.get("provider", "TensorRT inference artifact"),
            "nms_iou": args.nms_iou,
            "protocol": "offline merge of saved corner-window predictions, original-coordinate translation, one-class NMS",
            **metrics,
        }
        output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
