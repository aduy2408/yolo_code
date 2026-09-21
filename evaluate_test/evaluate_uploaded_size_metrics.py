#!/usr/bin/env python3
"""Backfill TinyBenchmark-compatible size metrics for uploaded native test splits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Support the documented direct invocation: ``python evaluate_test/<script>``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets
from train_scripts.train_all_yolo_baselines_no_mosaic import IMAGE_SIZES, local_ultralytics, prepare_dataset

MODELS = ("yolov5", "yolov8", "yolov9", "yolov10", "yolov11")
SEEDS = (42, 43, 44)
DATASETS = ("varroa", "levirship")
STANDARD_METRICS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--data-root", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def roots_from_args(args: argparse.Namespace) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in args.data_root:
        dataset, separator, path = value.partition("=")
        if dataset not in DATASETS or not separator or not path:
            raise ValueError(f"Invalid --data-root {value!r}")
        roots[dataset] = Path(path).resolve()
    missing = [dataset for dataset in DATASETS if dataset not in roots]
    if missing:
        raise ValueError(f"Missing dataset roots: {missing}")
    return roots


def evaluate_standard_metrics(
    run_dir: Path,
    data_yaml: Path,
    dataset: str,
    *,
    batch: int,
    device: str,
    workers: int,
) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=IMAGE_SIZES[dataset],
            batch=batch,
            device=device,
            workers=workers,
            iou=0.5,
            plots=False,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
    return metrics


def uploaded_prefixes(api: HfApi, repo_id: str) -> set[str]:
    return {
        path[: -len("/weights/best.pt")]
        for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        if path.endswith("/weights/best.pt") and any(path.startswith(f"runs/{dataset}/") for dataset in DATASETS)
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required evaluation")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    roots = roots_from_args(args)
    local_ultralytics()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    remote_files = set(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    prefixes = sorted(uploaded_prefixes(api, args.repo_id))
    remote_size_prefixes = {
        path[: -len("/size_metrics_complete.json")]
        for path in remote_files
        if path.endswith("/size_metrics_complete.json")
    }
    print(json.dumps({"repo_id": args.repo_id, "uploaded_native_runs": len(prefixes)}, sort_keys=True), flush=True)

    yaml_by_dataset = {
        dataset: prepare_dataset(dataset, roots[dataset], args.dataset_root)
        for dataset in DATASETS
    }
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[
                pattern
                for prefix in prefixes
                for pattern in (
                    f"{prefix}/weights/best.pt",
                    f"{prefix}/evaluation_metrics.json",
                    f"{prefix}/experiment_manifest.json",
                )
            ],
            local_dir=str(args.project / "hf_snapshot"),
            token=token,
        )
    )

    pending_operations: list[CommitOperationAdd] = []
    pending_prefixes: set[str] = set()
    for prefix in prefixes:
        _, dataset, model, seed_name = prefix.split("/")
        seed = int(seed_name.removeprefix("seed_"))
        run_dir = args.project / dataset / model / seed_name
        run_dir.mkdir(parents=True, exist_ok=True)
        source_weight = snapshot / prefix / "weights/best.pt"
        target_weight = run_dir / "weights/best.pt"
        target_weight.parent.mkdir(parents=True, exist_ok=True)
        if target_weight.exists() or target_weight.is_symlink():
            target_weight.unlink()
        target_weight.symlink_to(source_weight)
        metrics_path = run_dir / "evaluation_metrics.json"
        manifest_path = run_dir / "experiment_manifest.json"
        remote_prefix = prefix
        current_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        needs_standard = any(key not in current_metrics for key in STANDARD_METRICS)
        has_local_marker = metrics_path.exists() and (run_dir / "size_metrics_complete.json").exists()
        if not args.force and has_local_marker and not needs_standard:
            if remote_prefix in remote_size_prefixes:
                print(f"SKIP_SIZE_VERIFIED {remote_prefix}", flush=True)
                continue
            print(f"QUEUE_SIZE_UPLOAD {remote_prefix}", flush=True)
            pending_prefixes.add(remote_prefix)
            for local, remote in (
                (metrics_path, f"{remote_prefix}/evaluation_metrics.json"),
                (manifest_path, f"{remote_prefix}/experiment_manifest.json"),
                (run_dir / "evaluation/test_size_ground_truth.json", f"{remote_prefix}/evaluation/test_size_ground_truth.json"),
                (run_dir / "evaluation/test_size_predictions.json", f"{remote_prefix}/evaluation/test_size_predictions.json"),
                (run_dir / "size_metrics_complete.json", f"{remote_prefix}/size_metrics_complete.json"),
            ):
                pending_operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
            continue
        metrics = current_metrics
        if needs_standard:
            print(f"EVALUATE_STANDARD {remote_prefix}", flush=True)
            metrics.update(
                evaluate_standard_metrics(
                    run_dir,
                    yaml_by_dataset[dataset],
                    dataset,
                    batch=args.batch_size,
                    device=args.device,
                    workers=args.workers,
                )
            )
        if not has_local_marker:
            size_metrics = evaluate_native_test_size_buckets(
                run_dir,
                yaml_by_dataset[dataset],
                imgsz=IMAGE_SIZES[dataset],
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
            metrics.update(size_metrics)
            metrics["test_size/dataset"] = dataset
        else:
            size_metrics = {
                "test_size/protocol": metrics.get(
                    "test_size/protocol",
                    "TinyBenchmark area buckets on native YOLO test images; IoU=0.50:0.05:0.75; maxDets=200",
                )
            }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest.update({
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "split_seed": 42,
            "nms_iou": 0.5,
            "hf_repo_id": args.repo_id,
            **{key: metrics[key] for key in STANDARD_METRICS if key in metrics},
        })
        manifest["test_size_protocol"] = size_metrics["test_size/protocol"]
        manifest["test_size_source_artifact"] = "evaluation/test_size_predictions.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        marker = {
            "repo_id": args.repo_id,
            "remote_prefix": remote_prefix,
            "protocol": size_metrics["test_size/protocol"],
            "verified": [
                f"{remote_prefix}/evaluation_metrics.json",
                f"{remote_prefix}/experiment_manifest.json",
                f"{remote_prefix}/evaluation/test_size_ground_truth.json",
                f"{remote_prefix}/evaluation/test_size_predictions.json",
            ],
        }
        marker_path = run_dir / "size_metrics_complete.json"
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for local, remote in (
            (metrics_path, f"{remote_prefix}/evaluation_metrics.json"),
            (manifest_path, f"{remote_prefix}/experiment_manifest.json"),
            (run_dir / "evaluation/test_size_ground_truth.json", f"{remote_prefix}/evaluation/test_size_ground_truth.json"),
            (run_dir / "evaluation/test_size_predictions.json", f"{remote_prefix}/evaluation/test_size_predictions.json"),
            (marker_path, f"{remote_prefix}/size_metrics_complete.json"),
        ):
            pending_operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        pending_prefixes.add(remote_prefix)

    if pending_operations:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=pending_operations,
            commit_message="Add native test size-bucket metrics",
        )
        print(f"BATCH_UPLOADED {len(pending_prefixes)} runs", flush=True)

    for remote_prefix in sorted(pending_prefixes):
        remote_files = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
        missing = {
            f"{remote_prefix}/evaluation_metrics.json",
            f"{remote_prefix}/experiment_manifest.json",
            f"{remote_prefix}/evaluation/test_size_ground_truth.json",
            f"{remote_prefix}/evaluation/test_size_predictions.json",
            f"{remote_prefix}/size_metrics_complete.json",
        }
        if not missing.issubset(remote_files):
            raise RuntimeError(f"Remote size metric verification failed for {remote_prefix}: {sorted(missing - remote_files)}")
        print(f"SIZE_COMPLETE {remote_prefix}", flush=True)


if __name__ == "__main__":
    main()
