#!/usr/bin/env python3
"""Run one fixed-seed augmentation comparison across the three project datasets.

This wrapper intentionally owns a small, explicit comparison:

* datasets: LEVIR-Ship, TinyPerson, Varroa
* methods: all three OACP variants, all five Mosaic variants, and two
  Copy-Paste variants
* training seed: 42 only; split seed remains fixed at 42

The resulting queue has 25 valid jobs, not 30, because the project protocol
intentionally skips Mosaic for LEVIR-Ship. TinyPerson is fixed to one Mosaic
mode so it does not expand into duplicate mosaic/no-mosaic controls.

The underlying matrix runner still owns preparation, training, evaluation, and
HF upload. Real training must be launched through ``python -m utils.marimo_ops
launch``. This file only constructs the immutable command and refuses the old
augmentation defaults by always enabling ``--baseline-augmentation-parity``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from train_scripts.train_all_augmentation_matrix import (
    COPY_PASTE_CHOICES,
    MOSAIC_VARIANTS,
    OACP_VARIANTS,
    main as matrix_main,
)

DEFAULT_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_OACP_VARIANTS = tuple(OACP_VARIANTS)
DEFAULT_MOSAIC_VARIANTS = tuple(v for v in MOSAIC_VARIANTS if v != "M2_cluster_preserving")
DEFAULT_COPY_PASTE_VARIANTS = ("negative_canvas_r1", "negative_canvas_cp3_cluster1")


def _repo_id(value: str | None) -> str:
    repo_id = value or os.environ.get("MARIMO_HF_REPO_ID") or os.environ.get("HF_REPO_ID")
    if not repo_id or not repo_id.strip():
        raise SystemExit(
            "Missing task-specific HF repository. Pass --hf-repo-id or set MARIMO_HF_REPO_ID."
        )
    return repo_id.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", help="One task-specific dataset repo used for all three methods")
    parser.add_argument("--project", type=Path, default=Path("runs/augmentation_three_datasets_seed42"))
    parser.add_argument("--dataset-root", type=Path, default=Path("runs/augmentation_three_datasets_seed42/datasets"))
    parser.add_argument("--data-root-levir", type=Path, default=Path("/marimo/LevirShip/LevirShipData"))
    parser.add_argument("--data-root-tinyperson", type=Path, default=Path("/marimo/TinyPerson"))
    parser.add_argument("--data-root-varroa", type=Path, default=Path("/marimo/Varroa"))
    parser.add_argument("--scale-statistics-levir", type=Path)
    parser.add_argument("--scale-statistics-tinyperson", type=Path)
    parser.add_argument("--scale-statistics-varroa", type=Path)
    parser.add_argument("--hard-negative-bank-levir", type=Path)
    parser.add_argument("--hard-negative-bank-tinyperson", type=Path)
    parser.add_argument("--hard-negative-bank-varroa", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--oacp-variants", nargs="+", choices=OACP_VARIANTS,
        default=list(DEFAULT_OACP_VARIANTS),
    )
    parser.add_argument(
        "--mosaic-variants", nargs="+", choices=MOSAIC_VARIANTS,
        default=list(DEFAULT_MOSAIC_VARIANTS),
    )
    parser.add_argument(
        "--copy-paste-variants",
        nargs="+",
        choices=COPY_PASTE_CHOICES[1:],
        default=list(DEFAULT_COPY_PASTE_VARIANTS),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imgsz-levir", type=int, default=512)
    parser.add_argument("--imgsz-tinyperson", type=int, default=640)
    parser.add_argument("--imgsz-varroa", type=int, default=640)
    parser.add_argument("--tinyperson-mosaic-mode", choices=("mosaic", "no_mosaic"), default="mosaic")
    parser.add_argument("--range", dest="job_range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-effective-config", action="store_true")
    return parser.parse_args(argv)


def build_matrix_args(args: argparse.Namespace) -> list[str]:
    repo_id = _repo_id(args.hf_repo_id)
    if args.seed != DEFAULT_SEED:
        raise ValueError("This controlled rerun file is fixed to training seed 42")
    if args.split_seed != DEFAULT_SPLIT_SEED:
        raise ValueError("This controlled rerun file is fixed to split seed 42")
    if args.epochs != 100 or args.patience != 0 or args.workers != 8:
        raise ValueError("This controlled rerun requires epochs=100, patience=0, workers=8")

    output = [
        "--data-root-levir", str(args.data_root_levir),
        "--data-root-tinyperson", str(args.data_root_tinyperson),
        "--data-root-varroa", str(args.data_root_varroa),
        "--dataset-root", str(args.dataset_root),
        "--project", str(args.project),
        "--hf-repo-oacp", repo_id,
        "--hf-repo-mosaic", repo_id,
        "--hf-repo-copy-paste", repo_id,
        "--datasets", "levir", "tinyperson", "varroa",
        "--methods", "oacp", "mosaic", "copy_paste",
        "--oacp-variants", *args.oacp_variants,
        "--mosaic-variants", *args.mosaic_variants,
        "--copy-paste-variants", *args.copy_paste_variants,
        "--tinyperson-mosaic-modes", args.tinyperson_mosaic_mode,
        "--seeds", str(DEFAULT_SEED),
        "--split-seed", str(DEFAULT_SPLIT_SEED),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--device", args.device,
        "--amp" if args.amp else "--no-amp",
        "--imgsz-levir", str(args.imgsz_levir),
        "--imgsz-tinyperson", str(args.imgsz_tinyperson),
        "--imgsz-varroa", str(args.imgsz_varroa),
        "--baseline-augmentation-parity",
        "--confirm-settings",
    ]
    if args.job_range is not None:
        output.extend(["--range", str(args.job_range[0]), str(args.job_range[1])])
    for name in (
        "scale_statistics_levir", "scale_statistics_tinyperson", "scale_statistics_varroa",
        "hard_negative_bank_levir", "hard_negative_bank_tinyperson", "hard_negative_bank_varroa",
    ):
        value = getattr(args, name)
        if value:
            output.extend([f"--{name.replace('_', '-')}", str(value)])
    if args.prepare_only:
        output.append("--prepare-only")
    if args.print_effective_config:
        output.append("--print-effective-config")
    return output


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    matrix_main(build_matrix_args(args))


if __name__ == "__main__":
    main()
