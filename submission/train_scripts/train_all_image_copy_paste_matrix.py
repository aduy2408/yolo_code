#!/usr/bin/env python3
"""Train the fixed all-image Copy-Paste R2/R4 matrix.

The matrix covers YOLOv8, YOLOv9, and YOLOv11 on Levir, TinyPerson, and
Varroa with one seed. The two variants reuse the R2/R4 canvas policies but
apply them to every training image, not only original negative images.
With the fixed TinyPerson Mosaic mode, this expands to 18 queue jobs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_scripts.train_all_augmentation_matrix import main as run_matrix


DEFAULT_HF_REPO = "duyle2408/model-augmentation-seed42-all-image-cp-runs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--data-root-levir", type=Path, default=Path("/marimo/LevirShip/LevirShipData"))
    parser.add_argument("--data-root-tinyperson", type=Path, default=Path("/marimo/TinyPerson"))
    parser.add_argument("--data-root-varroa", type=Path, default=Path("/marimo/Varroa"))
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--range", dest="job_range", type=int, nargs=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    forwarded = [
        "--data-root-levir", str(args.data_root_levir),
        "--data-root-tinyperson", str(args.data_root_tinyperson),
        "--data-root-varroa", str(args.data_root_varroa),
        "--dataset-root", str(args.dataset_root),
        "--project", str(args.project),
        "--hf-repo-oacp", args.hf_repo,
        "--hf-repo-mosaic", args.hf_repo,
        "--hf-repo-copy-paste", args.hf_repo,
        "--datasets", "levir", "tinyperson", "varroa",
        "--models", "yolov8", "yolov9", "yolov11",
        "--methods", "copy_paste",
        "--copy-paste-variants", "all_canvas_r2", "all_canvas_r4",
        "--tinyperson-mosaic-modes", "mosaic",
        "--split-seed", "42",
        "--epochs", "100",
        "--patience", "0",
        "--workers", "8",
        "--batch-size", "8",
        "--nms-iou", "0.5",
        "--model-prefix",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
        "--confirm-settings",
    ]
    if args.job_range is not None:
        forwarded.extend(["--range", str(args.job_range[0]), str(args.job_range[1])])
    if args.print_effective_config:
        forwarded.append("--print-effective-config")
    if args.prepare_only:
        forwarded.append("--prepare-only")
    run_matrix(forwarded)


if __name__ == "__main__":
    main()
