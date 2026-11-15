"""Full-test paired context sensitivity metrics for baseline vs OACP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .rescue_scan import box_iou, clean_background, context_swap, match
from .common import load_yolo


def size_bucket(gt: list[float]) -> str:
    diagonal = float(np.hypot(gt[2] - gt[0], gt[3] - gt[1]))
    if diagonal < 8:
        return "tiny_lt8"
    if diagonal < 16:
        return "tiny_8_16"
    if diagonal < 32:
        return "small_16_32"
    return "medium_ge32"


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def summarize(rows: list[dict[str, Any]], swaps: int, include_size: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"objects": len(rows), "swaps_per_object": swaps, "models": {}}
    for model in ("baseline", "oacp"):
        original = [row[f"{model}_original"] for row in rows]
        swapped = [item for row in rows for item in row[f"{model}_swaps"]]
        original_recall = mean([float(item["hit"]) for item in original])
        swapped_recall = mean([float(item["hit"]) for item in swapped])
        original_iou = [float(item["iou"]) for item in original]
        paired_iou_drop = [float(row[f"{model}_original"]["iou"]) - mean([float(item["iou"]) for item in row[f"{model}_swaps"]]) for row in rows]
        original_scores = [float(item["score"]) for item in original]
        swapped_scores = [float(item["score"]) for item in swapped]
        result["models"][model] = {
            "original_recall_iou_ge_0_3": original_recall,
            "context_swap_recall_iou_ge_0_3": swapped_recall,
            "recall_drop": original_recall - swapped_recall,
            "mean_original_iou": mean(original_iou),
            "mean_paired_iou_drop": mean(paired_iou_drop),
            "median_paired_iou_drop": float(np.median(paired_iou_drop)) if paired_iou_drop else float("nan"),
            "mean_original_score": mean(original_scores),
            "mean_context_swap_score": mean(swapped_scores),
            "context_sensitivity_relative_score": mean([abs(float(row[f"{model}_original"]["score"]) - mean([float(item["score"]) for item in row[f"{model}_swaps"]])) / max(float(row[f"{model}_original"]["score"]), 1e-8) for row in rows]),
        }
    result["paired"] = {
        "both_original_hit": sum(row["baseline_original"]["hit"] and row["oacp_original"]["hit"] for row in rows),
        "baseline_swap_miss_oacp_swap_hit": sum(
            not swap["baseline"]["hit"] and swap["oacp"]["hit"]
            for row in rows
            for swap in row["paired_swaps"]
        ),
        "oacp_swap_miss_baseline_swap_hit": sum(
            not swap["oacp"]["hit"] and swap["baseline"]["hit"]
            for row in rows
            for swap in row["paired_swaps"]
        ),
    }
    if include_size:
        result["by_size"] = {}
        for bucket in sorted({row["size_bucket"] for row in rows}):
            subset = [row for row in rows if row["size_bucket"] == bucket]
            result["by_size"][bucket] = summarize(subset, swaps, include_size=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oacp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--swap-margin", type=int, default=2)
    parser.add_argument("--swaps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-object-size",
        type=float,
        default=None,
        help="Keep only GT objects with sqrt(box area) below this threshold.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.test_json.read_text())
    images = {item["id"]: item for item in manifest["images"]}
    image_ids = list(images)
    annotations: dict[int, list[list[float]]] = {}
    for item in manifest["annotations"]:
        annotations.setdefault(item["image_id"], []).append([float(item["bbox"][0]), float(item["bbox"][1]), float(item["bbox"][0] + item["bbox"][2]), float(item["bbox"][1] + item["bbox"][3])])
    paths = [args.data_root / item["file_name"] for item in images.values()]
    paths = [path if path.is_file() else Path("datasets/LevirShipData/All Images") / path.name for path in paths]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(next(str(path) for path in paths if not path.is_file()))
    baseline = load_yolo(args.baseline, args.device)
    oacp = load_yolo(args.oacp, args.device)
    all_annotations = annotations
    selected_annotations: dict[int, list[list[float]]] = {}
    if args.max_object_size is None:
        selected_annotations = annotations
    else:
        for image_id, boxes in annotations.items():
            eligible = [
                box for box in boxes
                if float(np.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1]))))
                < args.max_object_size
            ]
            if eligible:
                selected_annotations[image_id] = eligible
        annotations = selected_annotations
    selected_image_ids = [image_id for image_id in image_ids if image_id in annotations]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        batch_ids = image_ids[start : start + args.batch_size]
        frames = [cv2.imread(str(path)) for path in batch_paths]
        base_results = baseline.predict(frames, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        oacp_results = oacp.predict(frames, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
        batch_rows: list[dict[str, Any]] = []
        for local_index, image_id in enumerate(batch_ids):
            for object_id, gt in enumerate(annotations.get(image_id, [])):
                batch_rows.append({
                    "image": str(batch_paths[local_index]), "image_id": image_id, "object_id": object_id, "gt": gt,
                    "size_bucket": size_bucket(gt),
                    "baseline_original": match(base_results[local_index], gt, args.match_iou),
                    "oacp_original": match(oacp_results[local_index], gt, args.match_iou),
                    "baseline_swaps": [], "oacp_swaps": [], "paired_swaps": [],
                    "frame": frames[local_index], "source_index": start + local_index,
                })
        for offset in range(1, args.swaps + 1):
            if not batch_rows:
                break
            swapped_images = []
            for row in batch_rows:
                donor_index = (row["source_index"] + offset) % len(image_ids)
                donor_id = image_ids[donor_index]
                donor = cv2.imread(str(paths[donor_index]))
                swapped_images.append(context_swap(row["frame"], donor, row["gt"], all_annotations.get(donor_id, []), args.swap_margin))
            base_swap_results = baseline.predict(swapped_images, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
            oacp_swap_results = oacp.predict(swapped_images, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False, batch=args.batch_size)
            for row, base_swap_result, oacp_swap_result in zip(batch_rows, base_swap_results, oacp_swap_results):
                base_record = match(base_swap_result, row["gt"], args.match_iou)
                oacp_record = match(oacp_swap_result, row["gt"], args.match_iou)
                row["baseline_swaps"].append(base_record)
                row["oacp_swaps"].append(oacp_record)
                row["paired_swaps"].append({"baseline": base_record, "oacp": oacp_record})
        for row in batch_rows:
            row.pop("frame")
            row.pop("source_index")
        rows.extend(batch_rows)
        print(f"scanned {min(start + args.batch_size, len(paths))}/{len(paths)} images", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = {
        "protocol": {
            "test_json": str(args.test_json),
            "image_count": len(paths),
            "selected_image_count": len(selected_image_ids),
            "object_count": len(rows),
            "max_object_size": args.max_object_size,
            "size_definition": "sqrt((x2-x1)*(y2-y1))",
            "match_iou": args.match_iou,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "swap_margin": args.swap_margin,
            "swaps": args.swaps,
        },
        "summary": summarize(rows, args.swaps),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
