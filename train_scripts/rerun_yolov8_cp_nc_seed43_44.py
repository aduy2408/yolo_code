#!/usr/bin/env python3
"""Run the requested YOLOv8 copy-paste seed sweep.

The fixed queue is:

* datasets: LEVIR-Ship, TinyPerson, and Varroa
* variants: cp1_single1, cp2_single2, negative_canvas_r4, negative_canvas_r2
* training seeds: 43 and 44
* split seed: 42
* epochs/patience/workers/batch: 100/0/8/8
* detector: canonical YOLOv8n baseline configuration

This wrapper only assembles the matrix arguments. Real training must still be
preflighted and launched through ``python -m utils.marimo_ops launch``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_scripts.train_all_augmentation_matrix import main as matrix_main


FIXED_ARGS = [
    "--models", "yolov8",
    "--datasets", "levir", "tinyperson", "varroa",
    "--methods", "copy_paste",
    "--copy-paste-variants", "cp1_single1", "cp2_single2", "negative_canvas_r4", "negative_canvas_r2",
    "--seeds", "43", "44",
    "--split-seed", "42",
    "--epochs", "100",
    "--patience", "0",
    "--batch-size", "8",
    "--workers", "8",
    "--nms-iou", "0.5",
    "--tinyperson-mosaic-modes", "mosaic",
    "--baseline-augmentation-parity",
    "--include-varroa-negative-canvas",
    "--model-prefix",
    "--confirm-settings",
]


FORBIDDEN_OVERRIDES = {
    "--models", "--datasets", "--methods", "--copy-paste-variants",
    "--seed", "--seeds", "--split-seed", "--epochs", "--patience",
    "--batch-size", "--workers", "--nms-iou", "--baseline-augmentation-parity",
    "--no-baseline-augmentation-parity", "--tinyperson-mosaic-modes",
}


def _reject_fixed_overrides(argv: list[str]) -> None:
    conflicting = sorted(option for option in FORBIDDEN_OVERRIDES if option in argv)
    if conflicting:
        raise SystemExit("These settings are fixed by the seed sweep protocol: " + ", ".join(conflicting))


def main(argv: list[str] | None = None) -> None:
    passthrough = list(sys.argv[1:] if argv is None else argv)
    _reject_fixed_overrides(passthrough)
    matrix_main(FIXED_ARGS + passthrough)


if __name__ == "__main__":
    main()
