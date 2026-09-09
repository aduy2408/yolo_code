#!/usr/bin/env python3
"""Train TinyPerson YOLOv9t/YOLOv10n with legacy OACP and Mosaic.

This runner is intentionally separate from the standard baseline runner. It
forces the repository Ultralytics fork so OACP is available, enables the
historical two-call OACP path, and requires a reviewed settings preview before
any upload-required training launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
MODELS = {"yolov9": "yolov9t.pt", "yolov10": "yolov10n.pt"}
VARIANTS = tuple(MODELS)
STRICT_AUGMENTATION = {
    "context_augmentation": "oacp",
    "legacy_double_oacp": True,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "auto_augment": "randaugment",
    "erasing": 0.4,
}
TRAIN_AUGMENTATION = {
    key: value
    for key, value in STRICT_AUGMENTATION.items()
    if key not in {"context_augmentation", "legacy_double_oacp"}
}
SCHEDULE = {
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
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


def local_ultralytics():
    while str(ULTRALYTICS) in sys.path:
        sys.path.remove(str(ULTRALYTICS))
    sys.path.insert(0, str(ULTRALYTICS))
    from ultralytics import YOLO
    return YOLO


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_settings(args: argparse.Namespace, model_name: str, seed: int) -> dict:
    return {
        "model": model_name,
        "pretrained": MODELS[model_name],
        "seed": seed,
        "split_seed": args.split_seed,
        "data_root": str(args.data_root),
        "dataset_root": str(args.dataset_root),
        "project": str(args.project),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "amp": args.amp,
        "deterministic": True,
        "nms_iou": 0.5,
        "augmentation": dict(STRICT_AUGMENTATION),
        "schedule": dict(SCHEDULE),
        "upload_required": True,
        "hf_repo_id": args.hf_repo_id,
    }


def validate_settings(settings: dict) -> None:
    failures = []
    if settings["split_seed"] != 42:
        failures.append("split_seed must be 42")
    if settings["patience"] != 0:
        failures.append("patience must be 0")
    if settings["augmentation"]["context_augmentation"] != "oacp":
        failures.append("context_augmentation must be oacp")
    if not settings["augmentation"]["legacy_double_oacp"]:
        failures.append("legacy_double_oacp must be true")
    if settings["augmentation"]["mosaic"] != 1.0:
        failures.append("mosaic must be 1.0")
    if settings["augmentation"]["close_mosaic"] != 10:
        failures.append("close_mosaic must be 10")
    if failures:
        raise ValueError("Legacy OACP+Mosaic settings rejected: " + "; ".join(failures))


def print_effective_config(args: argparse.Namespace) -> None:
    runs = [effective_settings(args, model, seed) for seed in args.seeds for model in args.models]
    print(json.dumps({"confirmation_required": True, "runs": runs}, indent=2, sort_keys=True))


def complete(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED)


def train_one(model_name: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    os.environ["YOLO_CONTEXT_AUG"] = "oacp"
    os.environ["YOLO_LEGACY_DOUBLE_OACP"] = "1"
    os.environ["YOLO_VARIANT"] = f"tinyperson_{model_name}_legacy_oacp_mosaic"
    seed_everything(seed)
    YOLO = local_ultralytics()
    run_dir = args.project / model_name / f"seed_{seed}_corner_sw640_sh512"
    if complete(run_dir):
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODELS[model_name])
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch_size, device=args.device, workers=args.workers,
        patience=args.patience, seed=seed, deterministic=True, amp=args.amp,
        plots=False, project=str(args.project / model_name),
        name=f"seed_{seed}_corner_sw640_sh512", exist_ok=True,
        **TRAIN_AUGMENTATION,
        **SCHEDULE,
    )
    return run_dir


def write_metadata(model_name: str, seed: int, run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    config = {"model": MODELS[model_name], "name": model_name, "legacy_oacp": True, "mosaic": 1.0}
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n", encoding="utf-8")
    config_path = ROOT / "train_all_tinyperson_yolov9_yolov10_legacy_oacp_mosaic.py"
    manifest = effective_settings(args, model_name, seed)
    manifest.update({
        "data_yaml": str(data_yaml),
        "runner": config_path.name,
        "runner_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "required_artifacts": list(REQUIRED),
    })
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--confirm-settings", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    if args.print_effective_config:
        print_effective_config(args)
        return
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train. Review --print-effective-config, then add --confirm-settings.")
    for seed in args.seeds:
        for model in args.models:
            validate_settings(effective_settings(args, model, seed))
    from train_all_tinyperson import prepare_test_set, prepare_seed_dataset, evaluate
    from train_all_tinyperson_yolo_baselines import Uploader
    test_out = prepare_test_set(args.data_root, args.dataset_root)
    seed_dir = prepare_seed_dataset(args.data_root, args.dataset_root, test_out, args.split_seed)
    data_yaml = seed_dir / "tinyperson.yaml"
    uploader = Uploader(args.hf_repo_id)
    for seed in args.seeds:
        for model_name in args.models:
            run_dir = train_one(model_name, seed, data_yaml, args)
            evaluate(run_dir, data_yaml, test_out, args.data_root, args)
            write_metadata(model_name, seed, run_dir, data_yaml, args)
            if not complete(run_dir):
                raise RuntimeError(f"Incomplete artifacts: {run_dir}")
            uploader.upload_and_verify(model_name, seed, run_dir)
    print("TinyPerson YOLOv9/YOLOv10 legacy OACP+Mosaic matrix complete.", flush=True)


if __name__ == "__main__":
    main()
