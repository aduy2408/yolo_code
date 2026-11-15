#!/usr/bin/env python3
"""Run the three complementary LEVIR OACP combinations sequentially."""

from __future__ import annotations

import argparse
from pathlib import Path

from misc import train_context_aug_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
COMBINATIONS = {
    "dbss_oacp": (CONFIG_ROOT / "yolov8n_p2_levir_b_deep_supervision.yaml", "oacp", {"deep_sup": True}),
    "gap_oacp": (CONFIG_ROOT / "yolov8n_p2p3_levir_plain_gap.yaml", "oacp", {}),
    "gap_ftal_oacp": (CONFIG_ROOT / "yolov8n_p2p3_levir_plain_gap.yaml", "oacp", {"ftal": True}),
}


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
    parser.add_argument("--only", nargs="*", choices=sorted(COMBINATIONS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix.RUNS = COMBINATIONS
    args.only = list(COMBINATIONS) if args.only is None else args.only
    args.corrected_full_matrix = False
    args.corrected_api_ablation = False
    args.reuse_from = None
    args.reuse_variants = []
    args.amp = bool(args.amp)
    matrix.run(args)


if __name__ == "__main__":
    main()
