#!/usr/bin/env python3
"""Rerun the complete augmentation matrix with baseline YOLO augmentations.

This is the fixed protocol for the OACP, Mosaic, and Copy-Paste reruns. It
keeps the existing variant definitions, uses training seeds 42 and 43, fixes
the split seed at 42, and enables baseline augmentation parity:

* translate=0.1 and scale=0.5, as in the baseline runner's Ultralytics defaults;
* hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5;
* degrees, shear, perspective, flipud, mixup, and cutmix disabled unless the
  selected augmentation protocol explicitly changes Mosaic or Copy-Paste.

Operational paths, repositories, and Marimo launch flags are passed through to
``train_all_augmentation_matrix``. The fixed matrix and seed settings cannot be
overridden accidentally from the command line.

Real training must still be launched through ``python -m utils.marimo_ops launch``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_scripts.train_all_augmentation_matrix import main as matrix_main

FIXED_ARGS = [
    "--datasets", "levir", "tinyperson", "varroa",
    "--methods", "oacp", "mosaic", "copy_paste",
    "--oacp-variants", "load_adaptive", "spacing_adaptive", "mass_adaptive",
    "--mosaic-variants",
    "standard", "M2_cluster_preserving", "M3_post_scale_constrained",
    "M4_adaptive_geometry", "M5_hard_negative",
    "--copy-paste-variants",
    "cp1_single1", "cp3_cluster1", "negative_canvas_r1", "negative_canvas_r4",
    "--seeds", "42", "43",
    "--split-seed", "42",
    "--epochs", "100",
    "--patience", "0",
    "--workers", "8",
    "--baseline-augmentation-parity",
]

FORBIDDEN_OVERRIDES = {
    "--datasets", "--methods", "--oacp-variants", "--mosaic-variants",
    "--copy-paste-variants", "--seed", "--seeds", "--split-seed",
    "--baseline-augmentation-parity", "--no-baseline-augmentation-parity",
    "--epochs", "--patience", "--workers",
}


def _reject_fixed_overrides(argv: list[str]) -> None:
    conflicting = sorted(
        option for option in FORBIDDEN_OVERRIDES
        if option in argv
    )
    if conflicting:
        raise SystemExit(
            "These settings are fixed by the matched rerun protocol: "
            + ", ".join(conflicting)
        )


def main(argv: list[str] | None = None) -> None:
    passthrough = list(sys.argv[1:] if argv is None else argv)
    _reject_fixed_overrides(passthrough)
    matrix_main(FIXED_ARGS + passthrough)


if __name__ == "__main__":
    main()
