#!/usr/bin/env python3
"""Evaluate standard YOLO baselines on fixed LEVIR-Ship COCO splits.

The runner downloads only the requested ``best.pt`` checkpoints, predicts with
explicit NMS settings, and evaluates the resulting COCO detections with the
same AP/AP50/AP75/AP-small/AP-medium definitions used by MMDetection.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "duyle2408/levir-ship-yolo-baselines"
MODELS = ("yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n")
SEEDS = (42, 43, 44)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def checkpoint_filename(model: str, seed: int) -> str:
    return f"train/{model}_seed{seed}/weights/best.pt"


def precision(evaluator: Any, iou: float | None, area: str) -> float:
    import numpy as np

    values = evaluator.eval["precision"]
    if iou is not None:
        values = values[np.where(np.isclose(evaluator.params.iouThrs, iou))[0]]
    area_index = evaluator.params.areaRngLbl.index(area)
    values = values[:, :, :, area_index, -1]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def evaluate_coco(gt_path: Path, predictions: list[dict[str, Any]]) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(gt_path))
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map_50_95": precision(evaluator, None, "all"),
        "ap50": precision(evaluator, 0.50, "all"),
        "ap75": precision(evaluator, 0.75, "all"),
        "ap_small": precision(evaluator, None, "small"),
        "ap_medium": precision(evaluator, None, "medium"),
        "ap_large": precision(evaluator, None, "large"),
    }


def load_split(gt_path: Path, image_root: Path, limit: int = 0) -> list[dict[str, Any]]:
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    images = data["images"][:limit] if limit > 0 else data["images"]
    missing = [item["file_name"] for item in images if not (image_root / item["file_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return images


def predict_split(
    model: Any,
    images: list[dict[str, Any]],
    image_root: Path,
    *,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
) -> list[dict[str, Any]]:
    sources = [str(image_root / item["file_name"]) for item in images]
    results = model.predict(
        source=sources,
        stream=True,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        save=False,
    )
    detections: list[dict[str, Any]] = []
    for image, result in zip(images, results, strict=True):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        for (x1, y1, x2, y2), score in zip(xyxy, scores, strict=True):
            detections.append(
                {
                    "image_id": int(image["id"]),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return detections


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    output = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        item: dict[str, Any] = {"model": model, "seeds": [row["seed"] for row in model_rows]}
        for split in ("validation", "test"):
            item[split] = {}
            for metric in ("map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large"):
                values = [row[split][metric] for row in model_rows if row[split][metric] >= 0]
                item[split][metric] = {
                    "mean": statistics.mean(values) if values else -1.0,
                    "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        output.append(item)
    return output


def result_paths(models: Iterable[str], seeds: Iterable[int]) -> list[str]:
    paths = ["manifest.json", "results.json", "summary.json"]
    for model in models:
        for seed in seeds:
            root = f"{model}/seed_{seed}"
            paths.extend(
                (
                    f"{root}/metrics.json",
                    f"{root}/validation_predictions.json",
                    f"{root}/test_predictions.json",
                )
            )
    return paths


def upload_results(
    api: Any,
    *,
    repo_id: str,
    output_dir: Path,
    path_prefix: str,
    models: list[str],
    seeds: list[int],
) -> None:
    expected_local = result_paths(models, seeds)
    missing_local = [path for path in expected_local if not (output_dir / path).is_file()]
    if missing_local:
        raise RuntimeError(f"Refusing incomplete upload; missing: {missing_local}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        path_in_repo=path_prefix,
        ignore_patterns=["train.log", "train.pid", "state.json", "upload_complete.json"],
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected_remote = {f"{path_prefix}/{path}" for path in expected_local}
    missing_remote = sorted(expected_remote - remote_files)
    if missing_remote:
        raise RuntimeError(f"Hugging Face verification failed: {missing_remote}")
    marker = {
        "repo_id": repo_id,
        "path_prefix": path_prefix,
        "verified_files": sorted(expected_remote),
    }
    marker_path = output_dir / "upload_complete.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    marker_remote = f"{path_prefix}/{marker_path.name}"
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(marker_path),
        path_in_repo=marker_remote,
    )
    if marker_remote not in set(api.list_repo_files(repo_id, repo_type="dataset")):
        raise RuntimeError(f"Hugging Face marker verification failed: {marker_remote}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="Per-split smoke-test limit; 0 uses all images")
    parser.add_argument("--upload-results", action="store_true")
    parser.add_argument("--hf-path-prefix", default="evaluation/coco_fixedsplit42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.marimo_ops import require_training_context

    require_training_context(hf_repo_id=args.hf_repo_id)
    sys.path.insert(0, str(ROOT / "models_related" / "ultralytics"))
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO

    models = comma_list(args.models)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    seeds = [int(seed) for seed in comma_list(args.seeds)]
    splits = comma_list(args.splits)
    split_files = {"validation": "val.json", "test": "test.json"}
    if set(splits) - set(split_files):
        raise ValueError(f"Unknown splits: {sorted(set(splits) - set(split_files))}")

    for path in (args.coco_root, args.image_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    repo_info = HfApi(token=token).dataset_info(args.hf_repo_id)
    manifest = {
        "source_repo": args.hf_repo_id,
        "source_revision": repo_info.sha,
        "models": models,
        "seeds": seeds,
        "splits": splits,
        "coco_root": str(args.coco_root.resolve()),
        "image_root": str(args.image_root.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "limit": args.limit,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    split_data = {
        split: (
            args.coco_root / split_files[split],
            load_split(args.coco_root / split_files[split], args.image_root, args.limit),
        )
        for split in splits
    }
    rows: list[dict[str, Any]] = []
    total = len(models) * len(seeds)
    index = 0
    for model_name in models:
        for seed in seeds:
            index += 1
            run_dir = args.output_dir / model_name / f"seed_{seed}"
            final_path = run_dir / "metrics.json"
            if final_path.is_file():
                print(f"[{index}/{total}] SKIP verified metrics: {final_path}", flush=True)
                rows.append(json.loads(final_path.read_text(encoding="utf-8")))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            filename = checkpoint_filename(model_name, seed)
            print(f"[{index}/{total}] {model_name} seed {seed}: {filename}", flush=True)
            checkpoint = hf_hub_download(
                repo_id=args.hf_repo_id,
                filename=filename,
                repo_type="dataset",
                revision=repo_info.sha,
                token=token,
            )
            model = YOLO(checkpoint)
            row: dict[str, Any] = {
                "model": model_name,
                "seed": seed,
                "checkpoint": filename,
                "source_revision": repo_info.sha,
            }
            started = time.time()
            for split, (gt_path, images) in split_data.items():
                pred_path = run_dir / f"{split}_predictions.json"
                predictions = predict_split(
                    model,
                    images,
                    args.image_root,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                )
                pred_path.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
                row[split] = evaluate_coco(gt_path, predictions)
                row[split]["images"] = len(images)
                row[split]["detections"] = len(predictions)
            row["elapsed_seconds"] = time.time() - started
            final_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            del model

    rows.sort(key=lambda row: (row["model"], row["seed"]))
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = aggregate(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.upload_results:
        upload_results(
            HfApi(token=token),
            repo_id=args.hf_repo_id,
            output_dir=args.output_dir,
            path_prefix=args.hf_path_prefix.strip("/"),
            models=models,
            seeds=seeds,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
