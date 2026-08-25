#!/usr/bin/env python3
"""Train/evaluate/upload clean top-hat + GAP legacy Factorized TAL, without GGCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_all_levir_yolov8n_p2_tophat_gap_factorized_tal as topgap


ROOT = Path(__file__).resolve().parent
VARIANT = "tophat_gap_legacy_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_mode": "legacy",
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}

# Reuse the clean top-hat InputCueConv -> GAP ChannelAttention -> P2 Detect path.
topgap.VARIANT = VARIANT
topgap.SETTINGS = SETTINGS
topgap.gap.VARIANTS = {VARIANT: SETTINGS}
topgap.gap.DEFAULT_VARIANTS = (VARIANT,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_tophat_gap_legacy_factorized_tal_noggcf")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-tophat-gap-ftal-legacy-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = topgap.gap.Uploader(args.hf_repo_id)
    data_yaml = topgap.gap.base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = topgap.gap.train(variant, data_yaml, seed, args)
            topgap.gap.base.evaluate(run_dir, data_yaml, args)
            rows = topgap.gap.base.raw_p2_rows(run_dir, args)
            (run_dir / "factorized_tal_diagnostic.json").write_text(
                json.dumps(topgap.gap.base.diagnose_from_raw(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            topgap.gap.base.ranking_summary(rows, run_dir, args)
            topgap.gap.write_metadata(variant, run_dir, seed, args)
            if not topgap.gap.complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
