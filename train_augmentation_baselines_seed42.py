#!/usr/bin/env python3
"""Train the missing augmentation-reference baselines with seed 42.

This runner is intentionally limited to matched controls:

* canonical YOLOv8 P3/P4/P5-style detector, with no Mosaic or standard Mosaic;
* explicit YOLOv8n P2/P3/P4 plain detector, with no Mosaic or standard Mosaic;
* both LEVIR-Ship and TinyPerson;
* training seed 42 and fixed split seed 42.

The runner follows the recent Mosaic/Copy-Paste protocol: 100 epochs, patience
0, batch 8, workers 8, deterministic training, AMP, NMS IoU 0.50, sequential
train -> val/test evaluation -> task-specific HF upload verification. It must be
launched through ``python -m utils.marimo_ops launch`` for real training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related" / "ultralytics"
CANONICAL_CONFIG = ULTRALYTICS / "ultralytics/cfg/models/v8/yolov8.yaml"
CONFIGS = {
    "canonical": CANONICAL_CONFIG,
    "levir_p2p3p4": ROOT / "models_related/models_config/yolov8/levir/baseline_controls/yolov8n_p2p3p4_levir_plain.yaml",
    "tinyperson_p2p3p4": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3p4_plain.yaml",
}

# These are the only jobs this file owns. Keep the detector and augmentation
# policy explicit rather than deriving a baseline from a checkpoint filename.
JOBS = {
    "levir_canonical_no_aug_no_mosaic": ("levir", "canonical", "no_aug_no_mosaic"),
    "levir_canonical_standard_mosaic": ("levir", "canonical", "standard_mosaic"),
    "levir_p2p3p4_no_aug_no_mosaic": ("levir", "levir_p2p3p4", "no_aug_no_mosaic"),
    "levir_p2p3p4_standard_mosaic": ("levir", "levir_p2p3p4", "standard_mosaic"),
    "tinyperson_canonical_no_aug_no_mosaic": ("tinyperson", "canonical", "no_aug_no_mosaic"),
    "tinyperson_canonical_standard_mosaic": ("tinyperson", "canonical", "standard_mosaic"),
    "tinyperson_p2p3p4_no_aug_no_mosaic": ("tinyperson", "tinyperson_p2p3p4", "no_aug_no_mosaic"),
    "tinyperson_p2p3p4_standard_mosaic": ("tinyperson", "tinyperson_p2p3p4", "standard_mosaic"),
}

REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
)

# Match the recent Mosaic/Copy-Paste runs: disable competing geometric and
# object-mixing transforms, retain the ordinary Ultralytics color/flip policy.
COMMON_AUGMENTATION: dict[str, Any] = {
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "auto_augment": "randaugment",
    "erasing": 0.4,
}


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare_dataset(dataset: str, args: argparse.Namespace) -> tuple[Path, Path]:
    if dataset == "levir":
        from misc.prepare_levir_ship import prepare

        data_root = args.levir_data_root
        output = args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}"
        return prepare(data_root, output, args.split_seed), output

    import train_all_tinyperson as workflow

    data_root = args.tinyperson_data_root
    test_root = workflow.prepare_test_set(data_root, args.dataset_root)
    split_root = workflow.prepare_seed_dataset(data_root, args.dataset_root, test_root, args.split_seed)
    return split_root / "tinyperson.yaml", test_root


def augmentation_settings(policy: str) -> dict[str, Any]:
    if policy == "no_aug_no_mosaic":
        return {
            **COMMON_AUGMENTATION,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "mosaic_policy": "standard",
            "mosaic_postprocess_enabled": False,
        }
    if policy == "standard_mosaic":
        return {
            **COMMON_AUGMENTATION,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "mosaic_policy": "standard",
            "mosaic_postprocess_enabled": False,
        }
    raise ValueError(f"unknown baseline policy: {policy}")


def validate_config(config: Path, detector: str) -> None:
    if not config.is_file():
        raise FileNotFoundError(config)
    text = config.read_text(encoding="utf-8")
    if detector == "canonical":
        # The canonical YAML is the only accepted P3/P4/P5 control config.
        if config.resolve() != CANONICAL_CONFIG.resolve():
            raise ValueError(f"canonical baseline must use {CANONICAL_CONFIG}, got {config}")
        forbidden = ("CBAM", "ChannelAttention", "GAP")
        if any(token in text for token in forbidden):
            raise ValueError(f"canonical baseline contains an unrequested module: {config}")
    else:
        if "P2/P3/P4" not in text and "p2p3p4" not in config.name.lower():
            raise ValueError(f"expected explicit P2/P3/P4 config: {config}")


def training_complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    if not all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        return False
    return sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def manifest(job: str, dataset: str, detector: str, policy: str, data_yaml: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = CONFIGS[detector]
    return {
        "job": job,
        "dataset": dataset,
        "detector": detector,
        "model_config": str(config),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "variant": policy,
        "baseline_semantics": "no custom Mosaic/OACP/Copy-Paste; standard Ultralytics color/flip policy retained",
        "augmentation": augmentation_settings(policy),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "data_yaml": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz[dataset],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "amp": args.amp,
        "deterministic": True,
        "nms_iou": 0.5,
        "close_mosaic": 10 if policy == "standard_mosaic" else 0,
        "hf_repo_id": args.hf_repo_id[dataset],
        "commit_sha": git_sha(),
        "test_protocol": "standard Ultralytics test split; TinyPerson additionally records merged corner-window metrics",
    }


def train_one(job: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    dataset, detector, policy = JOBS[job]
    config = CONFIGS[detector]
    validate_config(config, detector)
    run_dir = args.project / job / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, run_dir / "config.yaml")
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest(job, dataset, detector, policy, data_yaml, args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if training_complete(run_dir, args.epochs):
        return run_dir

    local_ultralytics()
    from ultralytics import YOLO

    seed_everything(args.seed)
    model = YOLO(str(config), task="detect")
    model.load(args.pretrained, smart_transfer=True)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz[dataset],
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        seed=args.seed,
        deterministic=True,
        plots=False,
        project=str(args.project / job),
        name=f"seed_{args.seed}",
        exist_ok=True,
        **augmentation_settings(policy),
    )
    if not training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{job}: incomplete training artifacts")
    return run_dir


def evaluate_one(run_dir: Path, dataset: str, data_yaml: Path, test_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, Any] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz[dataset], batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        values = {key: float(value) for key, value in result.results_dict.items()}
        metrics.update({f"{split}/{key}": value for key, value in values.items()})
        metrics[f"{split}/AP50"] = values["metrics/mAP50(B)"]
        metrics[f"{split}/mAP50-95"] = values["metrics/mAP50-95(B)"]
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)

    if dataset == "tinyperson":
        import train_all_tinyperson as workflow

        custom_args = argparse.Namespace(
            imgsz=args.imgsz[dataset], batch_size=args.batch_size, device=args.device, workers=args.workers,
        )
        merged = workflow.evaluate_merged_test(run_dir, test_root, args.tinyperson_data_root, custom_args)
        metrics.update({key: float(value) for key, value in merged.items() if isinstance(value, (int, float))})
        metrics["test_protocol"] = "TinyPerson standard test plus merged corner-window evaluator"
        metrics["test_protocol_source"] = str(args.tinyperson_data_root / workflow.TEST_MERGED_JSON)
    else:
        metrics["test_protocol"] = "LEVIR-Ship standard held-out test split"
        metrics["test_protocol_source"] = str(data_yaml)

    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def upload_and_verify(run_dir: Path, dataset: str, job: str, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"{job}: refusing incomplete upload: {missing}")
    metrics = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    required_metrics = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
    if any(key not in metrics for key in required_metrics):
        raise RuntimeError(f"{job}: missing split-qualified metrics")
    repo_id = args.hf_repo_id[dataset]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    remote = f"runs/{job}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{path}" for path in REQUIRED}
    missing_remote = sorted(expected - remote_files)
    if missing_remote:
        raise RuntimeError(f"{job}: remote verification failed: {missing_remote}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levir-data-root", type=Path, required=True)
    parser.add_argument("--tinyperson-data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/augmentation_baselines_seed42")
    parser.add_argument("--hf-repo-levir", default="duyle2408/levir-augmentation-baselines-seed42-runs")
    parser.add_argument("--hf-repo-tinyperson", default="duyle2408/tinyperson-augmentation-baselines-seed42-runs")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--jobs", nargs="+", choices=tuple(JOBS), default=list(JOBS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imgsz-levir", type=int, default=512)
    parser.add_argument("--imgsz-tinyperson", type=int, default=640)
    parser.add_argument("--confirm-settings", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from utils.marimo_ops import require_training_context

    args = parse_args(argv)
    if args.seed != 42 or args.split_seed != 42 or args.epochs != 100 or args.patience != 0 or args.workers != 8:
        raise ValueError("This baseline matrix requires seed=42, split-seed=42, epochs=100, patience=0, workers=8")
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    require_training_context(hf_repo_id=args.hf_repo_levir)
    args.levir_data_root = args.levir_data_root.resolve()
    args.tinyperson_data_root = args.tinyperson_data_root.resolve()
    args.dataset_root, args.project = args.dataset_root.resolve(), args.project.resolve()
    args.hf_repo_id = {"levir": args.hf_repo_levir, "tinyperson": args.hf_repo_tinyperson}
    args.imgsz = {"levir": args.imgsz_levir, "tinyperson": args.imgsz_tinyperson}

    prepared: dict[str, tuple[Path, Path]] = {}
    for dataset in sorted({JOBS[job][0] for job in args.jobs}):
        prepared[dataset] = prepare_dataset(dataset, args)
    if args.prepare_only:
        for dataset, (data_yaml, _) in prepared.items():
            print(f"PREPARED {dataset}: {data_yaml}", flush=True)
        return

    for job in args.jobs:
        dataset = JOBS[job][0]
        data_yaml, test_root = prepared[dataset]
        run_dir = train_one(job, data_yaml, args)
        evaluate_one(run_dir, dataset, data_yaml, test_root, args)
        upload_and_verify(run_dir, dataset, job, args)
        print(f"COMPLETE {job}/seed_{args.seed}", flush=True)


if __name__ == "__main__":
    main()
