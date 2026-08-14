#!/usr/bin/env python3
"""Train, evaluate, and upload YOLOv8n LEVIR P2+P3 variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

EXPERIMENT = "levir_yolov8n_p2p3_experiments"
HF_REPO = "duyle2408/levir-yolov8n-p2p3-experiments"

VARIANTS = {
    "plain_p2p3": CONFIG_ROOT / "yolov8n_p2p3_levir_plain.yaml",
    "plain_p2p3_detail_repc2f": CONFIG_ROOT / "yolov8n_p2p3_levir_detail_repc2f.yaml",
    "plain_p2p3_detail_repc2f_factorized_k15": CONFIG_ROOT / "yolov8n_p2p3_levir_detail_repc2f.yaml",
    "plain_p2p3_detail_repc2f_gap_factorized_k15": CONFIG_ROOT / "yolov8n_p2p3_levir_detail_repc2f_gap.yaml",
    "plain_p2p3_gap_factorized_k15": CONFIG_ROOT / "yolov8n_p2p3_levir_plain_gap.yaml",
}

# Set the workflow globals
workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS

_base_train_kwargs = workflow.train_kwargs


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "_active_variant", "")
    if "factorized_k15" in variant:
        kwargs.update(
            factorized_tal_target=True,
            factorized_tal_tau=0.75,
            factorized_tal_kappa=1.5,
            factorized_tal_lambda=0.5,
            factorized_tal_s_max=32.0,
            factorized_tal_warmup_start=5,
            factorized_tal_warmup_end=15,
            factorized_tal_p2_only=True,
        )
    return kwargs


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    args._active_variant = variant
    return workflow.smoke(variant, data_yaml, args, amp)


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    args._active_variant = variant
    return workflow.train(variant, seed, data_yaml, amp, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16) # batch 16 as requested
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    workflow.train_kwargs = train_kwargs
    workflow.main = main
    workflow.parse_args = parse_args
    main()
