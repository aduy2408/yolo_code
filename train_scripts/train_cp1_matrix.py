#!/usr/bin/env python3
"""Train the fixed 9-run CP1 matrix: 3 detectors x 3 datasets.

This is a thin, explicit entry point over the canonical augmentation-matrix
runner. It keeps CP1, split provenance, detector YAMLs, and the task-specific
artifact repository fixed so the queue cannot silently expand to other
variants or TinyPerson mosaic modes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_scripts.train_all_augmentation_matrix import main as run_matrix


DEFAULT_HF_REPO = "duyle2408/model-augmentation-seed42-cp1-runs"


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
        "--copy-paste-variants", "cp1_single1",
        "--tinyperson-mosaic-modes", "mosaic",
        "--seeds", "42",
        "--split-seed", "42",
        "--epochs", "100",
        "--patience", "0",
        "--workers", "8",
        "--batch-size", "8",
        "--nms-iou", "0.5",
        "--model-prefix",
        "--confirm-settings",
    ]
    if args.print_effective_config:
        forwarded.append("--print-effective-config")
    if args.prepare_only:
        forwarded.append("--prepare-only")
    run_matrix(forwarded)


if __name__ == "__main__":
    main()
