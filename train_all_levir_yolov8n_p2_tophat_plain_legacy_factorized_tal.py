#!/usr/bin/env python3
"""Train/evaluate/upload clean top-hat + plain P2 legacy Factorized TAL, without GGCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_all_levir_yolov8n_p2_tophat_plain_factorized_tal as plain


ROOT = Path(__file__).resolve().parent
VARIANT = "tophat_plain_legacy_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_mode": "legacy",
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}

# Reuse the clean InputCueConv(top_hat) -> plain P2 Detect pipeline.
plain.VARIANT = VARIANT
plain.SETTINGS = SETTINGS
plain.plain.VARIANT = VARIANT
plain.plain.SETTINGS = SETTINGS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_tophat_plain_legacy_factorized_tal_noggcf")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-tophat-plain-ftal-legacy-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = plain.plain.Uploader(args.hf_repo_id)
    data_yaml = plain.plain.base.prepare_split(args)
    for seed in args.seeds:
        run_dir = plain.plain.train(VARIANT, data_yaml, seed, args)
        plain.plain.base.evaluate(run_dir, data_yaml, args)
        rows = plain.plain.base.raw_p2_rows(run_dir, args)
        (run_dir / "factorized_tal_diagnostic.json").write_text(
            json.dumps(plain.plain.base.diagnose_from_raw(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        plain.plain.base.ranking_summary(rows, run_dir, args)
        plain.plain.write_metadata(VARIANT, run_dir, seed, args)
        if not plain.plain.gap.complete(run_dir, args.epochs):
            raise RuntimeError(f"{VARIANT}: required post-evaluation artifacts are incomplete")
        uploader.upload_run(VARIANT, seed, run_dir)


if __name__ == "__main__":
    main()
