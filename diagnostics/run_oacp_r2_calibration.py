#!/usr/bin/env python3
"""Run a fixed-R2 train-split OACP calibration pass.

This is not model training. It loads only the prepared train split, applies the
same raw-sample OACP transform used by pre-transform training, and writes JSONL
records consumed by ``diagnostics/calibrate_oacp_effect.py``.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "models_related/ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from misc.prepare_levir_ship import prepare  # noqa: E402
from project_ultralytics.context_augment import OACP  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402


def configure_r2(diagnostics_path: Path) -> None:
    values = {
        "YOLO_CONTEXT_AUG": "oacp",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "OACP_PLACEMENT": "pre_transform",
        "OACP_PROB_POLICY": "fixed",
        "OACP_P": "0.40",
        "OACP_STRENGTH_POLICY": "fixed",
        "OACP_EFFECT_POLICY": "fixed",
        "OACP_PROTECTION_POLICY": "fixed",
        "OACP_STRENGTH_MIN": "0.10",
        "OACP_STRENGTH_MAX": "0.25",
        "OACP_SCALE_MIN": "0.80",
        "OACP_SCALE_MAX": "0.95",
        "OACP_PROTECTED_EXPAND": "3.0",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
        "OACP_DIAGNOSTICS_PATH": str(diagnostics_path),
    }
    os.environ.update(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configure_r2(args.output)
    yaml_path = prepare(args.data_root, args.dataset_root, args.split_seed)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    train_images = (yaml_path.parent / data["train"]).resolve()
    dataset = YOLODataset(
        img_path=str(train_images),
        imgsz=args.imgsz,
        augment=False,
        data=data,
        task="detect",
        cache=False,
        rect=False,
        batch_size=1,
        stride=32,
    )
    transform = OACP()
    for index in range(len(dataset)):
        transform(dataset.get_image_and_label(index))
        if (index + 1) % 250 == 0:
            print(f"calibrated {index + 1}/{len(dataset)}", flush=True)
    print(f"CALIBRATION_COMPLETE records={args.output} images={len(dataset)}", flush=True)


if __name__ == "__main__":
    main()
