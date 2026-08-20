#!/usr/bin/env python3
"""Train and evaluate the dataset-independent ORFS structure-only experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "levir_yolov8n_p2_orfs_structure"
CONFIG = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_orfs_structure.yaml"
VARIANTS = {"orfs_structure": CONFIG}

workflow.EXPERIMENT = EXPERIMENT
workflow.VARIANTS = VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=None)
    return parser.parse_args()


_base_train_kwargs = workflow.train_kwargs


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        orfs_gain=1.0,
        orfs_center_gain=1.0,
        orfs_geometry_gain=0.25,
        orfs_core_ratio=0.7,
        orfs_warmup_start=5,
        orfs_warmup_end=15,
    )
    return kwargs


def main() -> None:
    args = parse_args()
    args.variants = list(VARIANTS)
    args.runner = Path(__file__).resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    workflow.train_kwargs = train_kwargs

    data_yaml = workflow.prepare_fixed_split(args)
    amp = {variant: args.amp for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args, amp=args.amp) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)


if __name__ == "__main__":
    main()
