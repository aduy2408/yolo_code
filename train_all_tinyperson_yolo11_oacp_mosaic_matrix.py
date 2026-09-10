#!/usr/bin/env python3
"""Train the TinyPerson YOLO11n OACP/Mosaic four-variant matrix.

Variants compare the historical double-OACP path with default OACP parameters
against the single-pass new OACP path using the R2-frequent-mild parameters,
with Mosaic independently enabled and disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
MODEL = "yolo11n.pt"
DEFAULT_VARIANTS = (
    "legacy_default_oacp_mosaic",
    "legacy_default_oacp_no_mosaic",
    "r2_frequent_mild_oacp_mosaic",
    "r2_frequent_mild_oacp_no_mosaic",
)
VARIANTS = {
    "legacy_default_oacp_mosaic": {
        "mode": "oacp", "legacy_double_oacp": True, "mosaic": 1.0,
        "close_mosaic": 10, "oacp": {"p": 0.20, "expand": 3.0, "strength": [0.20, 0.40], "scale": [0.65, 0.85]},
    },
    "legacy_default_oacp_no_mosaic": {
        "mode": "oacp", "legacy_double_oacp": True, "mosaic": 0.0,
        "close_mosaic": 0, "oacp": {"p": 0.20, "expand": 3.0, "strength": [0.20, 0.40], "scale": [0.65, 0.85]},
    },
    "r2_frequent_mild_oacp_mosaic": {
        "mode": "oacp", "legacy_double_oacp": False, "mosaic": 1.0,
        "close_mosaic": 10, "oacp": {"p": 0.40, "expand": 3.0, "strength": [0.10, 0.25], "scale": [0.80, 0.95]},
    },
    "r2_frequent_mild_oacp_no_mosaic": {
        "mode": "oacp", "legacy_double_oacp": False, "mosaic": 0.0,
        "close_mosaic": 0, "oacp": {"p": 0.40, "expand": 3.0, "strength": [0.10, 0.25], "scale": [0.80, 0.95]},
    },
}
GEOMETRY_VARIANTS = {
    "current_oacp_mosaic": {
        "mode": "oacp", "legacy_double_oacp": False, "mosaic": 1.0,
        "close_mosaic": 10, "oacp_variant": "current",
        "oacp": {"p": 0.20, "expand": 3.0, "strength": [0.20, 0.40], "scale": [0.65, 0.85]},
    },
    "budget_oacp_mosaic": {
        "mode": "oacp", "legacy_double_oacp": False, "mosaic": 1.0,
        "close_mosaic": 10, "oacp_variant": "budget",
        "oacp": {"p": 0.20, "expand": 3.0, "strength": [0.20, 0.40], "scale": [0.65, 0.85]},
    },
    "density_oacp_mosaic": {
        "mode": "oacp", "legacy_double_oacp": False, "mosaic": 1.0,
        "close_mosaic": 10, "oacp_variant": "density",
        "oacp": {"p": 0.20, "expand": 3.0, "strength": [0.20, 0.40], "scale": [0.65, 0.85]},
    },
}
ALL_VARIANTS = {**VARIANTS, **GEOMETRY_VARIANTS}
TRAIN_AUGMENTATION = {
    "mixup": 0.0, "copy_paste": 0.0, "degrees": 0.0, "translate": 0.1,
    "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
    "fliplr": 0.5, "bgr": 0.0, "hsv_h": 0.015, "hsv_s": 0.7,
    "hsv_v": 0.4, "auto_augment": "randaugment", "erasing": 0.4,
}
SCHEDULE = {"optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
            "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1, "cos_lr": False}
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
            "evaluation_metrics.json", "config.yaml", "experiment_manifest.json")


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


def effective_settings(args: argparse.Namespace, variant: str, seed: int) -> dict:
    spec = ALL_VARIANTS[variant]
    return {
        "variant": variant, "model": MODEL, "pretrained": MODEL, "seed": seed,
        "split_seed": args.split_seed, "data_root": str(args.data_root),
        "dataset_root": str(args.dataset_root), "project": str(args.project),
        "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
        "batch_size": args.batch_size, "workers": args.workers, "device": args.device,
        "amp": args.amp, "deterministic": True, "nms_iou": 0.5,
        "augmentation": {**TRAIN_AUGMENTATION, **{k: spec[k] for k in ("mode", "legacy_double_oacp", "mosaic", "close_mosaic")},
                          "oacp_variant": spec.get("oacp_variant", "current"),
                          "oacp": dict(spec["oacp"])},
        "schedule": dict(SCHEDULE), "upload_required": True, "hf_repo_id": args.hf_repo_id,
    }


def validate_settings(settings: dict) -> None:
    if settings["split_seed"] != 42:
        raise ValueError("split_seed must remain fixed at 42")
    if settings["patience"] != 0:
        raise ValueError("patience must be 0")
    if settings["augmentation"]["mode"] != "oacp":
        raise ValueError("context augmentation must be oacp")
    if settings["augmentation"]["mosaic"] not in (0.0, 1.0):
        raise ValueError("mosaic must be 0.0 or 1.0")


def print_effective_config(args: argparse.Namespace) -> None:
    runs = [effective_settings(args, variant, seed) for seed in args.seeds for variant in args.variants]
    print(json.dumps({"confirmation_required": True, "runs": runs}, indent=2, sort_keys=True))


def complete(run_dir: Path) -> bool:
    return all((run_dir / item).is_file() for item in REQUIRED)


def configure_environment(variant: str) -> None:
    spec = ALL_VARIANTS[variant]
    oacp = spec["oacp"]
    os.environ.update({
        "YOLO_CONTEXT_AUG": "oacp",
        "YOLO_LEGACY_DOUBLE_OACP": "1" if spec["legacy_double_oacp"] else "0",
        "OACP_P": str(oacp["p"]), "OACP_PROTECTED_EXPAND": str(oacp["expand"]),
        "OACP_STRENGTH_MIN": str(oacp["strength"][0]), "OACP_STRENGTH_MAX": str(oacp["strength"][1]),
        "OACP_SCALE_MIN": str(oacp["scale"][0]), "OACP_SCALE_MAX": str(oacp["scale"][1]),
        "OACP_VARIANT": spec.get("oacp_variant", "current"),
        "OACP_SWEEP_LABEL": variant,
        "YOLO_VARIANT": f"tinyperson_yolo11_{variant}",
    })


def train_one(variant: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    configure_environment(variant)
    seed_everything(seed)
    YOLO = local_ultralytics()
    run_dir = args.project / variant / f"seed_{seed}_corner_sw640_sh512"
    if complete(run_dir):
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)
    spec = ALL_VARIANTS[variant]
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch_size, device=args.device, workers=args.workers,
                patience=args.patience, seed=seed, deterministic=True, amp=args.amp,
                plots=False, project=str(args.project / variant),
                name=f"seed_{seed}_corner_sw640_sh512", exist_ok=True,
                mosaic=spec["mosaic"], close_mosaic=spec["close_mosaic"],
                **TRAIN_AUGMENTATION, **SCHEDULE)
    return run_dir


def write_metadata(variant: str, seed: int, run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    path = ROOT / Path(__file__).name
    manifest = effective_settings(args, variant, seed)
    manifest.update({"data_yaml": str(data_yaml), "runner": path.name,
                     "runner_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "required_artifacts": list(REQUIRED), "command": sys.argv,
                     "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    (run_dir / "config.yaml").write_text(json.dumps({"model": MODEL, "variant": variant, **ALL_VARIANTS[variant]}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required")
        from huggingface_hub import HfApi
        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    def upload_and_verify(self, variant: str, seed: int, run_dir: Path) -> None:
        remote = f"runs/{variant}/seed_{seed}"
        expected = {f"{remote}/{item}" for item in REQUIRED}
        remote_files = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        if not (run_dir / "upload_complete.json").is_file() or not expected.issubset(remote_files):
            self.api.upload_folder(folder_path=str(run_dir), path_in_repo=remote,
                                   repo_id=self.repo_id, repo_type="dataset")
            remote_files = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        missing = sorted(expected - remote_files)
        if missing:
            raise RuntimeError(f"Upload verification failed for {variant}: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "variant": variant,
                                      "seed": seed, "remote_prefix": remote,
                                      "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json",
                             repo_id=self.repo_id, repo_type="dataset")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--variants", nargs="+", choices=tuple(ALL_VARIANTS), default=list(DEFAULT_VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-effective-config", action="store_true")
    p.add_argument("--confirm-settings", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    if args.print_effective_config:
        print_effective_config(args)
        return
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train. Review --print-effective-config, then add --confirm-settings.")
    for seed in args.seeds:
        for variant in args.variants:
            validate_settings(effective_settings(args, variant, seed))
    from train_all_tinyperson import prepare_test_set, prepare_seed_dataset, evaluate
    test_out = prepare_test_set(args.data_root, args.dataset_root)
    seed_dir = prepare_seed_dataset(args.data_root, args.dataset_root, test_out, args.split_seed)
    data_yaml = seed_dir / "tinyperson.yaml"
    uploader = Uploader(args.hf_repo_id)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train_one(variant, seed, data_yaml, args)
            evaluate(run_dir, data_yaml, test_out, args.data_root, args)
            write_metadata(variant, seed, run_dir, data_yaml, args)
            if not complete(run_dir):
                raise RuntimeError(f"Incomplete artifacts: {run_dir}")
            uploader.upload_and_verify(variant, seed, run_dir)
            print(f"COMPLETE {variant} seed={seed}", flush=True)
    print("TinyPerson YOLO11 OACP/Mosaic matrix complete.", flush=True)


if __name__ == "__main__":
    main()
