#!/usr/bin/env python3
"""Run the requested YOLO model and augmentation comparison at seed 42.

The queue is intentionally explicit:
- models: YOLOv8n, YOLOv9t, YOLO11n
- datasets: LEVIR-Ship, TinyPerson, Varroa
- methods: corrected single-pass OACP R2 and Copy-Paste
- Copy-Paste: cp2_single2, negative_canvas_r4, negative_canvas_r2
- split seed: 42
- training seed: 42

The underlying matrix runner owns dataset preparation, training, evaluation, and
Hugging Face upload. Launch this file only through ``python -m
utils.marimo_ops launch`` after the complete preflight passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_scripts.train_all_augmentation_matrix import main as matrix_main

FIXED_ARGS = [
    "--models", "yolov8", "yolov9", "yolov11",
    "--datasets", "levir", "tinyperson", "varroa",
    "--tinyperson-mosaic-modes", "mosaic",
    "--methods", "oacp", "copy_paste",
    "--oacp-variants", "r2",
    "--copy-paste-variants", "cp2_single2", "negative_canvas_r4", "negative_canvas_r2",
    "--seeds", "42",
    "--split-seed", "42",
    "--epochs", "100",
    "--patience", "0",
    "--workers", "8",
    "--baseline-augmentation-parity",
    "--model-prefix",
]

FORBIDDEN_OVERRIDES = {
    "--models", "--datasets", "--methods", "--oacp-variants", "--copy-paste-variants",
    "--seed", "--seeds", "--split-seed", "--baseline-augmentation-parity",
    "--no-baseline-augmentation-parity", "--epochs", "--patience", "--workers",
}


def _reject_fixed_overrides(argv: list[str]) -> None:
    conflicting = sorted(option for option in FORBIDDEN_OVERRIDES if option in argv)
    if conflicting:
        raise SystemExit("These settings are fixed by the model augmentation protocol: " + ", ".join(conflicting))


def main(argv: list[str] | None = None) -> None:
    passthrough = list(sys.argv[1:] if argv is None else argv)
    _reject_fixed_overrides(passthrough)
    matrix_main(FIXED_ARGS + passthrough)


if __name__ == "__main__":
    main()
