from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable

from checkpoint_compat import patch_legacy_modules
from ultralytics import YOLO
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyPerson windows with Ultralytics and TinyBenchmark COCOeval (mAP50-95).")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True, help="Bundle tinyperson_cocoeval.py")
    parser.add_argument("--predictions", type=Path, help="Reuse saved crop predictions and skip inference.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-only", action="store_true", help="Save crop predictions without merge or TinyBenchmark scoring.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def main() -> None:
    args = parse_args()
    data_root = args.data_root / "tinyperson"
    corner = json.loads((data_root / "annotations/corner/task/tiny_set_test_sw640_sh512_all.json").read_text())
    original_id = {row["file_name"]: int(row["id"]) for row in corner["old_images"]}
    records = sorted(corner["images"], key=lambda row: int(row["id"]))
    crop_predictions: list[dict] = []
    bundle_predictions: list[dict] = []
    if args.predictions is not None:
        crop_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        if not isinstance(crop_predictions, list):
            raise ValueError(f"Saved predictions must be a list: {args.predictions}")
        for det in crop_predictions:
            box = det.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"Invalid bbox_xyxy in saved prediction: {det}")
            x1, y1, x2, y2 = map(float, box)
            bundle_predictions.append({
                "image_id": int(det["crop_id"]),
                "category_id": 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(det["score"]),
            })
    else:
        model = YOLO(str(args.model), task="detect")
        patched_modules = patch_legacy_modules(model)
        if patched_modules:
            print(f"Applied inference compatibility defaults: {patched_modules}", flush=True)
        for row in tqdm(records, desc=f"TinyPerson {args.model.name}", unit="crop"):
            crop_id = int(row["id"])
            path = data_root / "images/all" / f"test_{crop_id:06d}.jpg"
            result = model.predict(
                source=str(path),
                imgsz=640,
                batch=1,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                half=args.precision == "fp16",
                verbose=False,
                save=False,
            )[0]
            original = original_id[row["file_name"]]
            dx, dy = row["corner"][:2]
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(float, box)
                score = float(score)
                crop_predictions.append({
                    "crop_id": crop_id,
                    "original_image_id": original,
                    "corner": [int(dx), int(dy)],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": score,
                })
                bundle_predictions.append({
                    "image_id": crop_id,
                    "category_id": 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": score,
                })

    args.results_root.mkdir(parents=True, exist_ok=True)
    base = args.model.stem
    corner_prediction_path = args.results_root / f"{base}_test_corner_predictions.json"
    corner_prediction_path.write_text(json.dumps(crop_predictions) + "\n", encoding="utf-8")
    metadata_path = args.results_root / f"{base}_test_corner_metadata.json"
    metadata_path.write_text(json.dumps({
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "dataset": "tinyperson",
        "split": "test_corner_windows",
        "provider": "Ultralytics YOLO.predict",
        "precision": args.precision,
        "image_size": 640,
        "batch_size": 1,
        "conf": args.conf,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "prediction_file": str(corner_prediction_path),
        "protocol": "Ultralytics corner-window prediction; bundle TinyBenchmark merge/scoring",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.save_only:
        print(json.dumps({"prediction_file": str(corner_prediction_path), "split": "test_corner_windows"}, sort_keys=True), flush=True)
        return

    by_original: dict[int, list[dict]] = defaultdict(list)
    corner_by_id = {int(image["id"]): image for image in corner["images"]}
    for det in bundle_predictions:
        crop_id = int(det["image_id"])
        row = corner_by_id.get(crop_id)
        if row is None:
            raise ValueError(f"Prediction image id {crop_id} is absent from corner annotation")
        name = row["file_name"]
        x, y, width, height = map(float, det["bbox"])
        x0, y0, _, _ = map(float, row["corner"])
        by_original[original_id[name]].append({
            "xyxy": [x + x0, y + y0, x + width + x0, y + height + y0],
            "score": float(det["score"]),
        })

    merged_predictions = []
    for image_id, candidates in by_original.items():
        merged_predictions.extend(_merge_group(image_id, candidates))

    merged_prediction_path = args.results_root / f"{base}_test_merged_predictions.json"
    merged_prediction_path.write_text(json.dumps(merged_predictions) + "\n", encoding="utf-8")
    merged_gt = COCO(str(data_root / "annotations/task/tiny_set_test_all.json"))
    coco_dt = merged_gt.loadRes(str(merged_prediction_path)) if merged_predictions else merged_gt.loadRes([])
    spec = importlib.util.spec_from_file_location("tinyperson_cocoeval", args.evaluator)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TinyBenchmark evaluator: {args.evaluator}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load TinyBenchmark evaluator: {args.evaluator}")
    spec.loader.exec_module(module)
    module.Params.EVAL_STRANDARD = "tiny"
    old_linspace = module.np.linspace
    had_np_float = hasattr(module.np, "float")
    if not had_np_float:
        module.np.float = float

    def compatible_linspace(start, stop, num=50, *positional, **keyword):
        if isinstance(num, (float, np.floating)):
            num = int(num)
        return old_linspace(start, stop, num, *positional, **keyword)

    module.np.linspace = compatible_linspace
    evaluator = module.COCOeval(merged_gt, coco_dt, "bbox", True, True)
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    try:
        evaluator.evaluate()
        evaluator.accumulate()
    finally:
        module.np.linspace = old_linspace
        if not had_np_float:
            delattr(module.np, "float")

    precision = evaluator.eval["precision"]
    labels = evaluator.params.areaRngLbl

    def mean_slice(iou_index: int, area_name: str) -> float | None:
        values = np.asarray(precision[iou_index, :, :, labels.index(area_name), -1])
        values = values[values > -1]
        return float(values.mean()) if values.size else None

    all_ap = [mean_slice(index, "all") for index in range(len(evaluator.params.iouThrs))]
    valid_ap = [value for value in all_ap if value is not None]
    ap75_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])
    metrics = {
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "dataset": "tinyperson",
        "split": "test_merged",
        "provider": "Ultralytics YOLO.predict",
        "precision": args.precision,
        "image_size": 640,
        "batch_size": 1,
        "conf": args.conf,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "prediction_file": str(merged_prediction_path),
        "evaluator": str(args.evaluator),
        "protocol": "TinyBenchmark merged original-image test; bundle corner mapping, class-agnostic NMS IoU=0.5, custom TinyBenchmark COCOeval IoU=0.50:0.05:0.95",
        "images_merged": len(merged_gt.dataset["images"]),
        "predictions_after_merge_nms": len(merged_predictions),
        "AP50": all_ap[0],
        "AP75": all_ap[ap75_index],
        "AP95": all_ap[-1],
        "mAP50-95": float(np.mean(valid_ap)) if valid_ap else None,
    }
    for area in ("tiny1", "tiny2", "tiny3", "tiny", "small", "medium"):
        if area in labels:
            metrics[f"AP50_{area}"] = mean_slice(0, area)
    metrics_path = args.results_root / f"{base}_test_merged_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)


def _merge_group(image_id: int, candidates: list[dict]) -> list[dict]:
    boxes = np.asarray([item["xyxy"] for item in candidates], dtype=np.float32)
    scores = np.asarray([item["score"] for item in candidates], dtype=np.float32)
    order = np.argsort(-scores, kind="mergesort")
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        current_box = boxes[current]
        candidate_boxes = boxes[remaining]
        x1 = np.maximum(current_box[0], candidate_boxes[:, 0])
        y1 = np.maximum(current_box[1], candidate_boxes[:, 1])
        x2 = np.minimum(current_box[2], candidate_boxes[:, 2])
        y2 = np.minimum(current_box[3], candidate_boxes[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        current_area = max(0.0, current_box[2] - current_box[0]) * max(0.0, current_box[3] - current_box[1])
        candidate_area = np.maximum(0.0, candidate_boxes[:, 2] - candidate_boxes[:, 0]) * np.maximum(0.0, candidate_boxes[:, 3] - candidate_boxes[:, 1])
        overlap = intersection / np.maximum(current_area + candidate_area - intersection, 1e-12)
        order = remaining[overlap <= 0.5]
    return [
        {
            "image_id": image_id,
            "category_id": 1,
            "bbox": [
                float(boxes[index, 0]),
                float(boxes[index, 1]),
                max(0.0, float(boxes[index, 2] - boxes[index, 0])),
                max(0.0, float(boxes[index, 3] - boxes[index, 1])),
            ],
            "score": float(scores[index]),
        }
        for index in kept
    ]


if __name__ == "__main__":
    main()
