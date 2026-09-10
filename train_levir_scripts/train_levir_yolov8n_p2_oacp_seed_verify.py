#!/usr/bin/env python3
"""Verify selected corrected single-pass OACP settings on additional seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from misc import train_context_aug_matrix as matrix
from train_levir_scripts.train_levir_yolov8n_p2_oacp_aggressive import (
    CONFIG,
    SCREEN,
    _configure,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-prefix", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    matrix.RUNS = {name: (CONFIG, "oacp", {}) for name in ("R2_frequent_mild", "R6_narrow_protection")}
    matrix.require_training_context(hf_repo_id=f"{a.hf_repo_prefix}-seed{a.seeds[0]}")
    for seed in a.seeds:
        project = a.project / f"seed{seed}"
        hf_repo = f"{a.hf_repo_prefix}-seed{seed}"
        for name in ("R2_frequent_mild", "R6_narrow_protection"):
            spec = dict(SCREEN["R2_frequent_mild"] if name.startswith("R2") else SCREEN["R4_strong"])
            if name.startswith("R6"):
                spec["expand"] = 1.5
            _configure(name, spec, selected_from="R4_strong" if name.startswith("R6") else "")
            args = argparse.Namespace(
                data_root=a.data_root,
                dataset_root=a.dataset_root,
                project=project,
                hf_repo_id=hf_repo,
                only=[name],
                corrected_api_ablation=False,
                corrected_full_matrix=False,
                reuse_from=None,
                reuse_variants=[],
                seed=seed,
                split_seed=a.split_seed,
                epochs=a.epochs,
                patience=a.patience,
                imgsz=a.imgsz,
                batch_size=a.batch_size,
                device=a.device,
                workers=a.workers,
                amp=a.amp,
                mosaic=0.0,
                close_mosaic=0,
                deterministic=a.deterministic,
            )
            matrix.run(args)
            print(f"COMPLETE seed={seed} variant={name}", flush=True)


if __name__ == "__main__":
    main()
