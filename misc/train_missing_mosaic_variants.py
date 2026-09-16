#!/usr/bin/env python3
"""Run the missing canonical YOLOv8 Mosaic variants for both datasets.

This is a thin sequential orchestrator around ``misc.train_mosaic_policy_matrix``.
It intentionally exposes only M2-M5 and never reruns a baseline or selects a
P2/P3/P4 detector. Launch this file through ``utils.marimo_ops launch``.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from misc import train_mosaic_policy_matrix as matrix

POLICIES = (
    "M2_cluster_preserving",
    "M3_post_scale_constrained",
    "M4_adaptive_geometry",
    "M5_hard_negative",
)
DATASETS = ("levir", "tinyperson")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levir-data-root", type=Path, required=True)
    parser.add_argument("--tinyperson-data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True, help="task-specific repository for this complete matrix")
    parser.add_argument("--levir-hard-negative-bank", type=Path, required=True)
    parser.add_argument("--tinyperson-hard-negative-bank", type=Path, required=True)
    parser.add_argument("--levir-scale-statistics", type=Path, required=True)
    parser.add_argument("--tinyperson-scale-statistics", type=Path, required=True)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--confirm-settings", action="store_true")
    return parser.parse_args(argv)


def _dataset_args(args: argparse.Namespace, dataset: str) -> Namespace:
    return Namespace(
        dataset=dataset,
        data_root=getattr(args, f"{dataset}_data_root"),
        dataset_root=args.dataset_root,
        project=args.project / dataset,
        hf_repo_id=args.hf_repo_id,
        pretrained=args.pretrained,
        policies=list(POLICIES),
        seed=args.seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=512 if dataset == "levir" else 640,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        close_mosaic=args.close_mosaic,
        hard_negative_bank=getattr(args, f"{dataset}_hard_negative_bank"),
        scale_statistics=getattr(args, f"{dataset}_scale_statistics"),
        confirm_settings=args.confirm_settings,
    )


def validate(args: argparse.Namespace) -> None:
    if not args.confirm_settings:
        raise RuntimeError("Refusing to train without --confirm-settings")
    if args.seed != 42 or args.split_seed != 42:
        raise ValueError("This matrix requires seed=42 and split-seed=42")
    if args.epochs != 100 or args.patience != 0:
        raise ValueError("This matrix requires epochs=100 and patience=0")
    for dataset in DATASETS:
        for field in ("hard_negative_bank", "scale_statistics"):
            path = getattr(args, f"{dataset}_{field}").resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{dataset} {field} does not exist: {path}")


def run_dataset(args: argparse.Namespace, dataset: str) -> None:
    run_args = _dataset_args(args, dataset)
    run_args.data_root = run_args.data_root.resolve()
    run_args.dataset_root = run_args.dataset_root.resolve()
    run_args.project = run_args.project.resolve()
    run_args.hard_negative_bank = run_args.hard_negative_bank.resolve()
    run_args.scale_statistics = run_args.scale_statistics.resolve()

    data_yaml, image_root, config = matrix.prepare_dataset(run_args)
    cache = matrix.build_context_cache(
        image_root,
        run_args.dataset_root / f"{dataset}_mosaic_context_seed{run_args.split_seed}.npz",
    )
    for policy in POLICIES:
        run_dir = run_args.project / policy / f"seed_{run_args.seed}"
        matrix.train(run_dir, data_yaml, config, cache, run_args, policy)
        matrix.evaluate(run_dir, data_yaml, run_args)
        matrix.upload_and_verify(run_dir, policy, run_args)
        print(f"COMPLETE {dataset}/{policy}/seed_{run_args.seed}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate(args)
    # The shared launcher injects MARIMO_TRAIN_WORKFLOW=1 and validates HF auth.
    matrix.require_training_context(hf_repo_id=args.hf_repo_id)
    for dataset in DATASETS:
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
