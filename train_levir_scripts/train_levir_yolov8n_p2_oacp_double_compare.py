#!/usr/bin/env python3
"""Run corrected single-pass OACP vs double-pass approximation on LEVIR-Ship.

The four matched variants are:
- oacp_standard_mosaic
- oacp_standard_no_mosaic
- oacp_double_approx_mosaic
- oacp_double_approx_no_mosaic

The approximation uses OACP_P=0.36, matching the probability that two
independent p=0.20 passes would affect a sample at least once. It is explicitly
an approximation, not a claim of pixel-identical double application.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from misc import train_context_aug_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"

VARIANTS = (
    ("oacp_standard_mosaic", "0.20", 1.0, 10),
    ("oacp_standard_no_mosaic", "0.20", 0.0, 0),
    ("oacp_double_approx_mosaic", "0.36", 1.0, 10),
    ("oacp_double_approx_no_mosaic", "0.36", 0.0, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["YOLO_CHECKPOINT"] = str(args.checkpoint)
    matrix.require_training_context(hf_repo_id=args.hf_repo_id)
    for name, probability, mosaic, close_mosaic in VARIANTS:
        os.environ["OACP_P"] = probability
        matrix.RUNS = {name: (CONFIG, "oacp", {})}
        matrix.run(argparse.Namespace(
            data_root=args.data_root,
            dataset_root=args.dataset_root,
            project=args.project,
            hf_repo_id=args.hf_repo_id,
            only=[name],
            corrected_api_ablation=False,
            corrected_full_matrix=False,
            reuse_from=None,
            reuse_variants=[],
            seed=args.seed,
            split_seed=args.split_seed,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            amp=args.amp,
            deterministic=args.deterministic,
            mosaic=mosaic,
            close_mosaic=close_mosaic,
        ))
        print(f"COMPLETE {name}", flush=True)


if __name__ == "__main__":
    main()
