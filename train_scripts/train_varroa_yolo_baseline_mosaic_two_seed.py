#!/usr/bin/env python3
"""Train and evaluate the five YOLO baselines on Varroa with Mosaic.

This is the dedicated two-seed recovery runner for the missing Varroa Mosaic
baseline slice. It reuses the baseline model construction, dataset preparation,
MuSGD compatibility patch, upload verification, and checkpoint layout from
``train_all_yolo_baselines_no_mosaic`` while changing only the augmentation
policy to standard Mosaic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from train_scripts import train_all_yolo_baselines_no_mosaic as base

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43)
SPLIT_SEED = 42
DATASET = "varroa"
MOSAIC = 1.0
CLOSE_MOSAIC = 10

REQUIRED_METRICS = (
    "val/AP50",
    "val/AP75",
    "val/mAP50-95",
    "val_size/AP50-Small",
    "val_size/AP75",
    "test/AP50",
    "test/AP75",
    "test/mAP50-95",
    "test_size/AP50-Small",
    "test_size/AP75",
)


def pinned_ultralytics() -> None:
    """Use the pinned upstream package, never the legacy compatibility fork."""
    upstream = str(ROOT / "vendor/ultralytics_upstream")
    sys.path[:] = [
        entry
        for entry in sys.path
        if "models_related/ultralytics" not in str(Path(entry).resolve())
    ]
    if upstream in sys.path:
        sys.path.remove(upstream)
    sys.path.insert(0, upstream)


base.local_ultralytics = pinned_ultralytics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/Varroa"))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/varroa_mosaic_two_seed")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_yolo_baselines_mosaic_two_seed")
    parser.add_argument("--models", nargs="+", choices=list(base.MODELS), default=list(base.MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--job-start", type=int, default=1, help="One-based inclusive job index in model-major order")
    parser.add_argument("--job-end", type=int, default=10, help="One-based inclusive job index in model-major order")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--hf-repo-id", required=True)
    return parser.parse_args(argv)


def evaluate_mosaic(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    pinned_ultralytics()
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import evaluate_native_size_buckets

    metrics: dict[str, float | str] = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml),
            split=split,
            imgsz=base.IMAGE_SIZES[DATASET],
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            iou=0.5,
            plots=False,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/AP75"] = float(result.box.map75)
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
        metrics.update(
            evaluate_native_size_buckets(
                run_dir,
                data_yaml,
                split=split,
                imgsz=base.IMAGE_SIZES[DATASET],
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(f"Missing required metrics: {missing}")
    metrics.update({"nms_iou": 0.5, "split_seed": args.split_seed, "test_protocol": "Native held-out Varroa test split"})
    return metrics


def train_one(model_name: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / DATASET / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not base.training_complete(run_dir):
        base.seed_everything(seed)
        model, _ = base.model_from_baseline_yaml(model_name)
        base.patch_musgd_noncontiguous()
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=base.IMAGE_SIZES[DATASET],
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=seed,
            deterministic=True,
            amp=True,
            optimizer=base.OPTIMIZER,
            mosaic=MOSAIC,
            close_mosaic=CLOSE_MOSAIC,
            save_period=10,
            plots=False,
            project=str(args.project / DATASET / model_name),
            name=f"seed_{seed}",
            exist_ok=True,
        )
    if not base.training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.split_seed != SPLIT_SEED:
        raise ValueError(f"This runner requires --split-seed {SPLIT_SEED}")
    if sorted(args.seeds) != sorted(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    jobs = [(model, seed) for model in args.models for seed in args.seeds]
    if not 1 <= args.job_start <= args.job_end <= len(jobs):
        raise ValueError(f"Job range must be within 1..{len(jobs)}")
    jobs = jobs[args.job_start - 1 : args.job_end]
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required training")

    from utils.marimo_ops import ensure_hf_repo, require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    token = os.environ["HF_TOKEN"]
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = base.prepare_dataset(DATASET, args.data_root, args.dataset_root)
    verified = base.verified_remote_prefixes(api, repo_id)
    print(json.dumps({"dataset": DATASET, "models": args.models, "seeds": args.seeds, "job_start": args.job_start, "job_end": args.job_end, "jobs": len(jobs), "mosaic": MOSAIC, "close_mosaic": CLOSE_MOSAIC, "split_seed": SPLIT_SEED}, sort_keys=True), flush=True)

    for model_name, seed in jobs:
        remote = f"runs/{DATASET}/{model_name}/seed_{seed}"
        if remote in verified:
            print(f"SKIP_VERIFIED {remote}", flush=True)
            continue
        run_dir = train_one(model_name, seed, data_yaml, args)
        metrics = evaluate_mosaic(run_dir, data_yaml, args)
        manifest = {
            "dataset": DATASET,
            "model": model_name,
            "pretrained": base.MODELS[model_name][0],
            "model_yaml": str((ROOT / base.MODELS[model_name][1]).resolve()),
            "seed": seed,
            "split_seed": SPLIT_SEED,
            "job_start": args.job_start,
            "job_end": args.job_end,
            "mosaic": MOSAIC,
            "close_mosaic": CLOSE_MOSAIC,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": base.IMAGE_SIZES[DATASET],
            "optimizer": base.OPTIMIZER,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "nms_iou": 0.5,
            "data_root": str(args.data_root.resolve()),
            "data_yaml": str(data_yaml),
            "git_sha": base.git_sha(),
            "hf_repo_id": repo_id,
            "required_metrics": list(REQUIRED_METRICS),
            "test_protocol": metrics["test_protocol"],
            **metrics,
        }
        (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        base.upload_and_verify(api, repo_id, run_dir, remote)
        print(f"COMPLETE {remote}", flush=True)


if __name__ == "__main__":
    main()
