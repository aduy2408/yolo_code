#!/usr/bin/env python3
"""Run matched YOLOv8s VisDrone baselines with explicit MuSGD or SGD.

The detector, dataset, augmentation, seed, and evaluation protocol are fixed.
Only the optimizer changes between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/ultralytics_upstream"
MODEL_WEIGHTS = "yolov8s.pt"
MODEL_YAML = ROOT / "vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml"
OPTIMIZERS = ("MuSGD", "SGD")
SEED = 42
IMAGE_SIZE = 640
MOSAIC = 1.0
CLOSE_MOSAIC = 10
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


def local_ultralytics() -> None:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/visdrone_yolov8s_optimizers")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/visdrone_yolov8s_optimizers")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--model-yaml", type=Path, default=MODEL_YAML)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-repo-id", required=True)
    return parser.parse_args(argv)


def prepare_dataset(data_root: Path, dataset_root: Path) -> Path:
    from train_scripts.train_all_visdrone_yolo_baselines import prepare_dataset as convert

    return convert(data_root.resolve(), dataset_root.resolve())


def model_from_yolov8s(model_yaml: Path):
    local_ultralytics()
    from ultralytics import YOLO

    alias_dir = ROOT / "runs/.baseline_yaml_aliases"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / "yolov8s.yaml"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(model_yaml.resolve())
    return YOLO(str(alias)).load(MODEL_WEIGHTS), alias


def train_one(optimizer: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    local_ultralytics()

    if optimizer == "MuSGD":
        from train_scripts.train_all_visdrone_yolo_baselines import patch_musgd_noncontiguous

        patch_musgd_noncontiguous()
    seed_everything(args.seed)
    run_dir = args.project / optimizer.lower() / "mosaic" / f"seed_{args.seed}"
    training_artifacts = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")
    if all((run_dir / item).is_file() for item in training_artifacts):
        return run_dir
    model, _ = model_from_yolov8s(args.model_yaml)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=IMAGE_SIZE,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        amp=True,
        optimizer=optimizer,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        plots=False,
        project=str(args.project / optimizer.lower() / "mosaic"),
        name=f"seed_{args.seed}",
        exist_ok=True,
    )
    if not all((run_dir / item).is_file() for item in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        raise RuntimeError(f"Training artifacts are incomplete: {run_dir}")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets
    from train_scripts.train_all_visdrone_yolo_baselines import metric_value

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, object] = {"nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=IMAGE_SIZE,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            iou=0.5,
            plots=False,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = metric_value(result, "metrics/mAP50(B)")
        metrics[f"{split}/mAP50-95"] = metric_value(result, "metrics/mAP50-95(B)")
    try:
        from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets

        metrics.update(
            evaluate_native_test_size_buckets(
                run_dir,
                data_yaml,
                imgsz=IMAGE_SIZE,
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
            )
        )
        metrics["test_size/dataset"] = "visdrone"
    except ModuleNotFoundError as exc:
        if exc.name != "pycocotools":
            raise
        metrics["test_size/protocol"] = "not_run: pycocotools unavailable"
    return metrics


def upload_and_verify(api: object, repo_id: str, run_dir: Path, remote: str) -> None:
    missing = [item for item in REQUIRED if not (run_dir / item).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload: {missing}")
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{item}" for item in REQUIRED}
    if not expected.issubset(files):
        raise RuntimeError(f"Upload verification failed: {sorted(expected - files)}")
    marker = run_dir / "upload_complete.json"
    verified = sorted(expected | {f"{remote}/upload_complete.json"})
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": remote, "verified": verified}, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required training")
    from utils.marimo_ops import ensure_hf_repo, require_training_context
    from huggingface_hub import HfApi

    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    api = HfApi(token=os.environ["HF_TOKEN"])
    data_yaml = prepare_dataset(args.data_root, args.dataset_root)
    print(json.dumps({"optimizers": args.optimizers, "repo_id": repo_id, "data_yaml": str(data_yaml)}, sort_keys=True), flush=True)

    for optimizer in args.optimizers:
        remote = f"runs/yolov8s/{optimizer.lower()}/mosaic/seed_{args.seed}"
        run_dir = train_one(optimizer, data_yaml, args)
        metrics = evaluate(run_dir, data_yaml, args)
        manifest = {
            "dataset": "VisDrone2019-DET",
            "data_root": str(args.data_root),
            "dataset_yaml": str(data_yaml),
            "official_split": "VisDrone2019-DET-train / val / test-dev",
            "split_provenance": "official VisDrone2019-DET split, no random reassignment",
            "model": "yolov8s",
            "pretrained": MODEL_WEIGHTS,
            "model_yaml": str(args.model_yaml.resolve()),
            "optimizer": optimizer,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "augmentation": "mosaic",
            "mosaic": MOSAIC,
            "close_mosaic": CLOSE_MOSAIC,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": IMAGE_SIZE,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "nms_iou": 0.5,
            "git_sha": git_sha(),
            "hf_repo_id": repo_id,
            "test_protocol": "Ultralytics native VisDrone2019-DET-test-dev split",
            "test_size_protocol": metrics.get("test_size/protocol", "native size-bucket evaluator"),
            **metrics,
        }
        (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        upload_and_verify(api, repo_id, run_dir, remote)
        print(f"COMPLETE {remote}", flush=True)


if __name__ == "__main__":
    main()
