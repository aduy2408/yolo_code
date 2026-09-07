#!/usr/bin/env python3
"""Sequentially run verified YOLOv8n P2 + OACP runs for seeds 43 and 44."""

from __future__ import annotations

import argparse
from pathlib import Path

from misc import train_context_aug_matrix as matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--hf-repo-prefix", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from utils.marimo_ops import require_training_context

    for seed in args.seeds:
        hf_repo = f"{args.hf_repo_prefix}-seed{seed}"
        require_training_context(hf_repo_id=hf_repo)
        run_args = argparse.Namespace(
            data_root=args.data_root,
            dataset_root=args.dataset_root / f"seed_{seed}",
            project=args.project / f"seed_{seed}",
            hf_repo_id=hf_repo,
            only=["baseline_oacp"],
            corrected_api_ablation=False,
            corrected_full_matrix=True,
            reuse_from=None,
            reuse_variants=[],
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            amp=args.amp,
        )
        matrix.run(run_args)
        print(f"COMPLETE seed_{seed}", flush=True)


if __name__ == "__main__":
    run(parse_args())
