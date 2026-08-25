#!/usr/bin/env python3
"""Reproduce the historical seed-42 GAP + Factorized TAL winner exactly."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_factorized_tal as gap


ROOT = Path(__file__).resolve().parent
VARIANT = "gap_factorized_legacy_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_mode": "legacy",
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}

# Keep the canonical GAP topology and make the historical FTAL semantic explicit.
gap.VARIANTS = {VARIANT: SETTINGS}
gap.DEFAULT_VARIANTS = (VARIANT,)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal_legacy")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-gap-factorized-tal-legacy-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = gap.Uploader(args.hf_repo_id)
    data_yaml = gap.base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = gap.train(variant, data_yaml, seed, args)
            gap.base.evaluate(run_dir, data_yaml, args)
            rows = gap.base.raw_p2_rows(run_dir, args)
            diag = gap.base.diagnose_from_raw(rows)
            (run_dir / "factorized_tal_diagnostic.json").write_text(
                __import__("json").dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            gap.base.ranking_summary(rows, run_dir, args)
            gap.write_metadata(variant, run_dir, seed, args)
            if not gap.complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
