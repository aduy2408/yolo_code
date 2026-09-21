#!/usr/bin/env python3
"""Train the canonical YOLOv8 OACP matrix for LEVIR-Ship and TinyPerson.

The detector is intentionally fixed to the upstream-style P3/P4/P5 YOLOv8
configuration.  LEVIR runs use no Mosaic; TinyPerson runs use standard Mosaic.
Each run is evaluated on val and test and uploaded before the next run starts.
Real training must be launched through ``python -m utils.marimo_ops launch``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related" / "ultralytics"
CANONICAL_CONFIG = ULTRALYTICS / "ultralytics/cfg/models/v8/yolov8.yaml"

from train_scripts.train_augmentation_baselines_seed42 import (  # noqa: E402
    COMMON_AUGMENTATION,
    REQUIRED,
    evaluate_one,
    git_sha,
    local_ultralytics,
    prepare_dataset,
    seed_everything,
    upload_and_verify,
)

VARIANTS = (
    "mass_adaptive",
    "load_adaptive",
    "size_adaptive",
    "effect_adaptive",
    "hardness_adaptive",
    "c1_strength_adaptive",
    "c2_scale_adaptive",
    "c3_contrast_adaptive",
)
CONTEXT_VARIANTS = {"c1_strength_adaptive", "c2_scale_adaptive", "c3_contrast_adaptive"}


def _job_names() -> tuple[str, ...]:
    return tuple(f"{dataset}_{variant}" for dataset in ("levir", "tinyperson") for variant in VARIANTS)


def _check_canonical_config() -> None:
    if not CANONICAL_CONFIG.is_file():
        raise FileNotFoundError(CANONICAL_CONFIG)
    text = CANONICAL_CONFIG.read_text(encoding="utf-8")
    if any(token in text for token in ("CBAM", "ChannelAttention", "GAP")):
        raise ValueError("canonical YOLOv8 config contains an unrequested custom module")


def _clear_oacp_env() -> None:
    for key in (
        "YOLO_CONTEXT_AUG", "YOLO_LEGACY_DOUBLE_OACP", "OACP_PROFILE", "OACP_VARIANT",
        "OACP_PLACEMENT", "OACP_PROB_POLICY", "OACP_P", "OACP_EFFECT_POLICY",
        "OACP_TARGET_EFFECT", "OACP_STRENGTH_POLICY", "OACP_SCALE_POLICY",
        "OACP_PROTECTION_POLICY", "OACP_CONTEXT_STATS_PATH", "OACP_P_MIN", "OACP_P_MAX",
        "OACP_STRENGTH_MIN", "OACP_STRENGTH_MAX", "OACP_SCALE_MIN", "OACP_SCALE_MAX",
    ):
        os.environ.pop(key, None)


def configure_variant(variant: str, context_stats: Path | None, effect_target: float) -> dict[str, str]:
    _clear_oacp_env()
    env = {
        "YOLO_CONTEXT_AUG": "oacp",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
        "OACP_VARIANT": "current",
        "OACP_PLACEMENT": "pre_transform",
        "OACP_PROB_POLICY": "fixed",
        "OACP_P": "0.20",
        "OACP_STRENGTH_MIN": "0.20",
        "OACP_STRENGTH_MAX": "0.40",
        "OACP_SCALE_MIN": "0.65",
        "OACP_SCALE_MAX": "0.85",
        "OACP_EFFECT_POLICY": "fixed",
        "OACP_STRENGTH_POLICY": "fixed",
        "OACP_SCALE_POLICY": "fixed",
        "OACP_PROTECTION_POLICY": "fixed",
    }
    if variant == "mass_adaptive":
        env["OACP_VARIANT"] = "mass_adaptive"
    elif variant == "load_adaptive":
        env["OACP_VARIANT"] = "load_adaptive"
    elif variant == "size_adaptive":
        env["OACP_PROTECTION_POLICY"] = "size_adaptive"
        env.update({"OACP_SIZE_EXPAND_MIN": "2.0", "OACP_SIZE_EXPAND_MAX": "4.0", "OACP_SIZE_EXPAND_SMAX": "32.0"})
    elif variant == "effect_adaptive":
        if effect_target <= 0:
            raise ValueError("--effect-target must be positive for effect_adaptive")
        env.update({
            "OACP_EFFECT_POLICY": "adaptive",
            "OACP_STRENGTH_POLICY": "effect_adaptive",
            "OACP_TARGET_EFFECT": str(effect_target),
        })
    elif variant == "hardness_adaptive":
        env["OACP_STRENGTH_POLICY"] = "hardness_adaptive"
    elif variant == "c1_strength_adaptive":
        env["OACP_STRENGTH_POLICY"] = "context_adaptive"
    elif variant == "c2_scale_adaptive":
        env["OACP_SCALE_POLICY"] = "context_adaptive"
    elif variant == "c3_contrast_adaptive":
        env["OACP_PROTECTION_POLICY"] = "contrast_adaptive"
    else:
        raise ValueError(f"unknown OACP variant: {variant}")
    if variant in CONTEXT_VARIANTS:
        if context_stats is None or not context_stats.is_file():
            raise FileNotFoundError(f"context statistics required for {variant}: {context_stats}")
        env["OACP_CONTEXT_STATS_PATH"] = str(context_stats)
    os.environ.update(env)
    return env


def _augmentation(dataset: str) -> dict[str, Any]:
    if dataset == "levir":
        return {**COMMON_AUGMENTATION, "mosaic": 0.0, "close_mosaic": 0, "mosaic_policy": "none"}
    return {**COMMON_AUGMENTATION, "mosaic": 1.0, "close_mosaic": 10, "mosaic_policy": "standard"}


def _training_complete(run_dir: Path, epochs: int) -> bool:
    if not all((run_dir / item).is_file() for item in ("weights/best.pt", "weights/last.pt", "results.csv")):
        return False
    results = run_dir / "results.csv"
    return sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def _manifest(job: str, dataset: str, variant: str, data_yaml: Path, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment": "yolov8_default_oacp_matrix",
        "job": job,
        "dataset": dataset,
        "variant": variant,
        "model_config": str(CANONICAL_CONFIG),
        "detector": "canonical YOLOv8 P3/P4/P5",
        "data_yaml": str(data_yaml),
        "augmentation": _augmentation(dataset),
        "oacp_env": {k: v for k, v in env.items() if k != "HF_TOKEN"},
        "seed": args.seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz[dataset],
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "amp": args.amp,
        "deterministic": True,
        "nms_iou": 0.5,
        "hf_repo_id": args.hf_repo_id[dataset],
        "commit_sha": git_sha(),
        "test_protocol": "LEVIR-Ship standard held-out test split" if dataset == "levir" else "TinyPerson standard test plus merged corner-window evaluator",
    }


def train_one(job: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    dataset, variant = job.split("/", 1)
    run_dir = args.project / dataset / variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = configure_variant(variant, args.context_stats[dataset], args.effect_target)
    (run_dir / "config.yaml").write_text(CANONICAL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(_manifest(job, dataset, variant, data_yaml, args, env), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _training_complete(run_dir, args.epochs):
        return run_dir
    local_ultralytics()
    from ultralytics import YOLO

    seed_everything(args.seed)
    model = YOLO(str(CANONICAL_CONFIG), task="detect")
    model.load(args.pretrained, smart_transfer=True)
    model.train(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz[dataset], batch=args.batch_size, device=args.device,
        workers=args.workers, amp=args.amp, seed=args.seed, deterministic=True,
        plots=False, project=str(args.project / dataset / variant), name=f"seed_{args.seed}",
        exist_ok=True, **_augmentation(dataset),
    )
    if not _training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{job}: incomplete training artifacts")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levir-data-root", type=Path, required=True)
    p.add_argument("--tinyperson-data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, default=ROOT / "runs/yolov8_default_oacp_matrix")
    p.add_argument("--hf-repo-levir", default="duyle2408/levir-yolov8-default-oacp-matrix-runs")
    p.add_argument("--hf-repo-tinyperson", default="duyle2408/tinyperson-yolov8-default-oacp-matrix-runs")
    p.add_argument("--context-stats-levir", type=Path)
    p.add_argument("--context-stats-tinyperson", type=Path)
    p.add_argument("--effect-target", type=float, default=2.0)
    p.add_argument("--pretrained", default="yolov8n.pt")
    p.add_argument("--jobs", nargs="+", choices=_job_names(), default=list(_job_names()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imgsz-levir", type=int, default=512)
    p.add_argument("--imgsz-tinyperson", type=int, default=640)
    p.add_argument("--confirm-settings", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from utils.marimo_ops import require_training_context

    args = parse_args(argv)
    if not args.confirm_settings:
        raise RuntimeError("refusing to train without --confirm-settings")
    if (args.seed, args.split_seed, args.epochs, args.patience, args.workers) != (42, 42, 100, 0, 8):
        raise ValueError("matrix requires seed=42, split-seed=42, epochs=100, patience=0, workers=8")
    _check_canonical_config()
    require_training_context(hf_repo_id=args.hf_repo_levir)
    require_training_context(hf_repo_id=args.hf_repo_tinyperson)
    from utils.marimo_ops import ensure_hf_repo
    ensure_hf_repo(args.hf_repo_levir)
    ensure_hf_repo(args.hf_repo_tinyperson)
    args.levir_data_root = args.levir_data_root.resolve()
    args.tinyperson_data_root = args.tinyperson_data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.context_stats = {"levir": args.context_stats_levir, "tinyperson": args.context_stats_tinyperson}
    args.hf_repo_id = {"levir": args.hf_repo_levir, "tinyperson": args.hf_repo_tinyperson}
    args.imgsz = {"levir": args.imgsz_levir, "tinyperson": args.imgsz_tinyperson}
    selected = [(job.split("_", 1)[0], job.split("_", 1)[1]) for job in args.jobs]
    if any(variant in CONTEXT_VARIANTS and args.context_stats[dataset] is None for dataset, variant in selected):
        raise ValueError("C1/C2/C3 jobs require --context-stats-levir and/or --context-stats-tinyperson")
    prepared = {dataset: prepare_dataset(dataset, args) for dataset in sorted({dataset for dataset, _ in selected})}
    if args.prepare_only:
        for dataset, (data_yaml, _) in prepared.items():
            print(f"PREPARED {dataset}: {data_yaml}", flush=True)
        return
    for dataset, variant in selected:
        job = f"{dataset}/{variant}"
        data_yaml, test_root = prepared[dataset]
        run_dir = train_one(job, data_yaml, args)
        evaluate_one(run_dir, dataset, data_yaml, test_root, args)
        upload_and_verify(run_dir, dataset, f"{dataset}_{variant}", args)
        print(f"COMPLETE {job}/seed_{args.seed}", flush=True)


if __name__ == "__main__":
    main()
