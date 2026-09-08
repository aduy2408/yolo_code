#!/usr/bin/env python3
"""Train base YOLOv8 P2 + OACP no-Mosaic with q/qmax_GT regression weighting."""

from __future__ import annotations

import argparse
from pathlib import Path

from misc import train_context_aug_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"
VARIANT = "yolov8n_p2_oacp_r1_qmax_no_mosaic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix.RUNS = {VARIANT: (CONFIG, "oacp", {"reg_weight_mode": "qmax_gt"})}
    matrix.run(argparse.Namespace(
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        project=args.project,
        hf_repo_id=args.hf_repo_id,
        only=[VARIANT],
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
        mosaic=0.0,
        close_mosaic=0,
    ))


if __name__ == "__main__":
    main()
