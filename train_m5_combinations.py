#!/usr/bin/env python3
"""Run M5 hard-negative Mosaic combinations on YOLOv8n P2/P3/P4.

Jobs are intentionally fixed to two requested combinations:
- LEVIR-Ship: M5 hard-negative Mosaic + OACP R2
- TinyPerson: M5 hard-negative Mosaic + clustered Copy-Paste CP3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train_p2p4_hard_negative_mosaic as base
from copy_paste_protocol import variant_overrides

JOBS = {
    "levir_m5_oacp_r2": "levir",
    "tinyperson_m5_cp3": "tinyperson",
}
REQUIRED = base.REQUIRED


def clear_augmentation_env() -> None:
    for key in (
        "YOLO_CONTEXT_AUG", "OACP_PROFILE", "OACP_VARIANT", "YOLO_LEGACY_DOUBLE_OACP",
        "OACP_SWEEP_LABEL",
    ):
        os.environ.pop(key, None)


def configure_job(job: str) -> dict[str, Any]:
    clear_augmentation_env()
    if job == "levir_m5_oacp_r2":
        os.environ.update({
            "YOLO_CONTEXT_AUG": "oacp",
            "OACP_PROFILE": "r2",
            "OACP_VARIANT": "current",
            "YOLO_LEGACY_DOUBLE_OACP": "0",
            "OACP_SWEEP_LABEL": "m5_oacp_r2",
        })
        return {
            "name": job,
            "dataset": "levir",
            "oacp": True,
            "oacp_profile": "r2",
            "copy_paste": False,
            "copy_paste_variant": None,
            "copy_paste_settings": {},
        }
    if job == "tinyperson_m5_cp3":
        settings = variant_overrides("cp3_cluster1")
        return {
            "name": job,
            "dataset": "tinyperson",
            "oacp": False,
            "oacp_profile": None,
            "copy_paste": True,
            "copy_paste_variant": "cp3_cluster1",
            "copy_paste_settings": settings,
        }
    raise ValueError(job)


def manifest(job: str, data_yaml: Path, bank: Path, args: argparse.Namespace) -> dict[str, Any]:
    spec = configure_job(job)
    dataset = spec["dataset"]
    config = base.CONFIGS[dataset]
    return {
        "experiment": job,
        "dataset": dataset,
        "detector": "YOLOv8n explicit P2/P3/P4 plain neck",
        "model_config": str(config),
        "model_config_sha256": __import__("hashlib").sha256(config.read_bytes()).hexdigest(),
        "data_yaml": str(data_yaml),
        "hard_negative_bank": str(bank),
        "augmentation": {
            "mosaic": 1.0,
            "close_mosaic": 10,
            "mosaic_policy": "hard_negative",
            "hard_negative_tile": True,
            "hardneg_mosaic_prob": 0.30,
            "oacp": spec["oacp"],
            "oacp_profile": spec["oacp_profile"],
            "copy_paste": spec["copy_paste"],
            "copy_paste_variant": spec["copy_paste_variant"],
            "copy_paste_settings": spec["copy_paste_settings"],
            "mixup": 0.0,
        },
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz[dataset],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "nms_iou": 0.5,
        "hf_repo_id": args.hf_repo_id[dataset],
        "commit_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "test_protocol": "LEVIR-Ship standard held-out test split" if dataset == "levir" else "TinyPerson standard test plus merged corner-window evaluator",
    }


def train_one(job: str, data_yaml: Path, train_images: Path, test_root: Path, args: argparse.Namespace) -> Path:
    spec = configure_job(job)
    dataset = spec["dataset"]
    run_dir = args.project / job / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = train_images.parent.parent / "labels/train"
    bank = run_dir / "hard_negative_bank.json"
    base.mine_bank(train_images, labels, bank, args.pretrained)
    config = base.CONFIGS[dataset]
    (run_dir / "config.yaml").write_bytes(config.read_bytes())
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest(job, data_yaml, bank, args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if all((run_dir / p).is_file() for p in REQUIRED if p != "evaluation_metrics.json"):
        return run_dir

    if str(base.ULTRA) not in sys.path:
        sys.path.insert(0, str(base.ULTRA))
    from ultralytics import YOLO

    base.seed_everything(args.seed)
    model = YOLO(str(config), task="detect")
    model.load(args.pretrained, smart_transfer=True)
    kwargs: dict[str, Any] = {
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mosaic_policy": "hard_negative",
        "hard_negative_tile": True,
        "hardneg_mosaic_prob": 0.30,
        "hard_negative_bank": str(bank),
        "mosaic_postprocess_enabled": False,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    if spec["copy_paste"]:
        # Reuse CP3 object-placement settings, but keep this experiment's M5
        # Mosaic gate and schedule instead of inheriting CP-only mosaic=0.
        kwargs.update({
            key: value
            for key, value in spec["copy_paste_settings"].items()
            if key not in {"mosaic", "close_mosaic", "mixup", "cutmix"}
        })
    model.train(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz[dataset], batch=args.batch_size, device=args.device,
        workers=args.workers, amp=args.amp, seed=args.seed, deterministic=True,
        plots=False, project=str(run_dir.parent), name=f"seed_{args.seed}", exist_ok=True,
        **kwargs,
    )
    if not all((run_dir / p).is_file() for p in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        raise RuntimeError(f"incomplete training artifacts: {run_dir}")
    return run_dir


def run_job(job: str, args: argparse.Namespace, prepared: dict[str, tuple[Path, Path, Path]]) -> None:
    spec = configure_job(job)
    data_yaml, train_images, test_root = prepared[spec["dataset"]]
    run_dir = train_one(job, data_yaml, train_images, test_root, args)
    clear_augmentation_env()
    base.evaluate(spec["dataset"], run_dir, data_yaml, test_root, args)
    base.upload_verify(spec["dataset"], run_dir, args)
    print(f"COMPLETE {job}/seed_{args.seed}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levir-data-root", type=Path, required=True)
    p.add_argument("--tinyperson-data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-levir", required=True)
    p.add_argument("--hf-repo-tinyperson", required=True)
    p.add_argument("--pretrained", default="yolov8n.pt")
    p.add_argument("--jobs", nargs="+", choices=tuple(JOBS), default=tuple(JOBS))
    p.add_argument("--seed", type=int, default=42); p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imgsz-levir", type=int, default=512); p.add_argument("--imgsz-tinyperson", type=int, default=640)
    p.add_argument("--confirm-settings", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from utils.marimo_ops import require_training_context
    args = parse_args(argv)
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    if (args.seed, args.split_seed, args.epochs, args.patience, args.workers) != (42, 42, 100, 0, 8):
        raise ValueError("requires seed=42, split-seed=42, epochs=100, patience=0, workers=8")
    require_training_context(hf_repo_id=args.hf_repo_levir)
    args.levir_data_root = args.levir_data_root.resolve(); args.tinyperson_data_root = args.tinyperson_data_root.resolve()
    args.dataset_root = args.dataset_root.resolve(); args.project = args.project.resolve()
    args.hf_repo_id = {"levir": args.hf_repo_levir, "tinyperson": args.hf_repo_tinyperson}
    args.imgsz = {"levir": args.imgsz_levir, "tinyperson": args.imgsz_tinyperson}
    prepared: dict[str, tuple[Path, Path, Path]] = {}
    for dataset in sorted({JOBS[job] for job in args.jobs}):
        prepared[dataset] = base.prepare_dataset(dataset, args)
    for job in args.jobs:
        run_job(job, args, prepared)


if __name__ == "__main__":
    main()
