#!/usr/bin/env python3
"""Matched TinyPerson B0/B1/B2 P2-slot experiment runner.

The runner is upload-required and intended to be launched only through
``python -m utils.marimo_ops launch`` in the live Marimo environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.marimo_ops import ensure_hf_repo, require_training_context

CONFIGS = {
    "B0": ROOT / "project_ultralytics/configs/p2_slots/yolov8n_tinyperson_p2p3p4_b0.yaml",
    "B1": ROOT / "project_ultralytics/configs/p2_slots/yolov8n_tinyperson_p2p3p4_b1_capacity.yaml",
    "B2": ROOT / "project_ultralytics/configs/p2_slots/yolov8n_tinyperson_p2p3p4_b2_demix.yaml",
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "args.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=sorted(CONFIGS), default=["B0", "B1", "B2"])
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    from train_scripts.train_all_tinyperson import prepare_seed_dataset, prepare_test_set

    test_root = prepare_test_set(args.data_root, args.dataset_root)
    data_yaml = prepare_seed_dataset(args.data_root, args.dataset_root, test_root, args.split_seed)
    return (data_yaml / "tinyperson.yaml").resolve(), test_root.resolve()


def load_model(config: Path):
    from project_ultralytics import load_project_model

    return load_project_model(config, task="detect", verbose=False)


def training_complete(run_dir: Path) -> bool:
    if not all((run_dir / item).is_file() for item in REQUIRED if item != "evaluation_metrics.json"):
        return False
    manifest_path = run_dir / "experiment_manifest.json"
    results_path = run_dir / "results.csv"
    if not manifest_path.is_file() or results_path.stat().st_size == 0:
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("amp") is not False:
            return False
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or int(float(rows[-1]["epoch"])) < int(manifest["epochs"]):
            return False
        for row in rows:
            for value in row.values():
                if value is None or value.strip().lower() in {"nan", "inf", "-inf"}:
                    return False
                if value.strip() and not math.isfinite(float(value)):
                    return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def train_one(args: argparse.Namespace, data_yaml: Path, variant: str, seed: int) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIGS[variant]
    manifest = {
        "experiment": "tinyperson_p2_slot_demixer",
        "variant": variant,
        "hypothesis": "P2 local representation demixing versus extra local prediction capacity",
        "seed": seed,
        "split_seed": args.split_seed,
        "data_root": str(args.data_root),
        "dataset_yaml": str(data_yaml),
        "model_yaml": str(config),
        "model_yaml_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "commit_sha": git_sha(),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "nms_iou": 0.5,
        "amp": False,
        "p2_slots": 2 if variant != "B0" else 1,
        "p2_variant": {"B0": "baseline", "B1": "capacity", "B2": "demix"}[variant],
        "mosaic": 0.0,
        "hf_repo_id": args.hf_repo_id,
        "test_protocol": "TinyPerson prepared test split from official corner-window annotations; val and test are evaluated separately.",
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if training_complete(run_dir):
        return run_dir

    seed_everything(seed)
    model = load_model(config)
    model.load(args.pretrained)
    from project_ultralytics.training import train_with_loss_adapter

    loss_adapter = "upstream" if variant == "B0" else "p2_slots"
    train_with_loss_adapter(
        model,
        loss_adapter=loss_adapter,
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        seed=seed,
        deterministic=True,
        amp=False,
        plots=False,
        project=str(args.project / variant),
        name=f"seed_{seed}",
        exist_ok=True,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        iou=0.5,
    )
    if not all((run_dir / item).is_file() for item in REQUIRED if item != "evaluation_metrics.json"):
        raise RuntimeError(f"Incomplete training artifacts for {variant} seed {seed}: {run_dir}")
    return run_dir


def evaluate_one(args: argparse.Namespace, data_yaml: Path, variant: str, seed: int, run_dir: Path) -> dict[str, float | str]:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        cached = json.loads(output.read_text())
        if all(key in cached for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")) and cached.get("commit_sha") == git_sha() and cached.get("amp") is False:
            return cached

    model = load_model(CONFIGS[variant])
    model.load(run_dir / "weights/best.pt")
    metrics: dict[str, float | str] = {
        "checkpoint": "best.pt",
        "commit_sha": git_sha(),
        "amp": False,
        "variant": variant,
        "seed": seed,
        "split_seed": args.split_seed,
        "nms_iou": 0.5,
        "test_protocol": "TinyPerson prepared test split from official corner-window annotations",
        "test_source_artifact": str(data_yaml),
    }
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            workers=args.workers,
            device=args.device,
            plots=False,
            iou=0.5,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        prefix = split
        metrics[f"{prefix}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{prefix}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
        metrics[f"{prefix}/AP75"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def upload_and_verify(args: argparse.Namespace, variant: str, seed: int, run_dir: Path) -> None:
    missing = [item for item in REQUIRED if not (run_dir / item).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {variant} seed {seed}: {missing}")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    remote = f"runs/{variant}/seed_{seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset")
    remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    expected = {f"{remote}/{item}" for item in REQUIRED}
    missing = sorted(expected - remote_files)
    if missing:
        raise RuntimeError(f"Remote upload verification failed for {variant} seed {seed}: {missing}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=args.hf_repo_id, repo_type="dataset")


def main() -> None:
    args = parse_args()
    require_training_context(hf_repo_id=args.hf_repo_id)
    ensure_hf_repo(args.hf_repo_id)
    if args.workers != 8:
        raise ValueError("This matched experiment requires workers=8")
    data_yaml, _ = prepare_dataset(args)
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = train_one(args, data_yaml, variant, seed)
            metrics = evaluate_one(args, data_yaml, variant, seed, run_dir)
            if not all(key in metrics for key in ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")):
                raise RuntimeError(f"Missing split-qualified metrics for {variant} seed {seed}")
            upload_and_verify(args, variant, seed, run_dir)
            print(f"completed variant={variant} seed={seed} val/AP50={metrics['val/AP50']:.6f} test/AP50={metrics['test/AP50']:.6f}", flush=True)


if __name__ == "__main__":
    main()
