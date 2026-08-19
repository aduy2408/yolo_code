#!/usr/bin/env python3
"""Train, evaluate, and upload seed-42 YOLOv8n P2 KVCA encoder + FTAL (oldnorm vs newnorm) experiments."""

import os
import sys
import json
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_all_levir_yolov8n_p2_routing as workflow

EXPERIMENT = "levir_yolov8n_p2_kvca_encoder_ftal"
HF_REPO = "duyle2408/levir-yolov8n-p2-kvca-encoder-ftal-seed42"
CONFIG = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_kvca_encoder_groupweight.yaml"

VARIANTS = {
    "kvca_encoder_ftal_oldnorm": CONFIG,
    "kvca_encoder_ftal_newnorm": CONFIG,
}

workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    return parser.parse_args()

_base_train_kwargs = workflow.train_kwargs

def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    
    # Configure hyperparameters for FTAL
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

_base_train = workflow.train

def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    # Set the appropriate FTAL_NORM_MODE environment variable dynamically
    if "oldnorm" in variant:
        os.environ["FTAL_NORM_MODE"] = "oldnorm"
        print(f"\n>>> [FTAL NORM] Setting FTAL_NORM_MODE = oldnorm for {variant} <<<\n", flush=True)
    elif "newnorm" in variant:
        os.environ["FTAL_NORM_MODE"] = "newnorm"
        print(f"\n>>> [FTAL NORM] Setting FTAL_NORM_MODE = newnorm for {variant} <<<\n", flush=True)
    return _base_train(variant, seed, data_yaml, amp, args)

def main() -> None:
    args = parse_args()
    args.variants = list(VARIANTS)
    args.runner = Path(__file__).resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    
    # Override workflow configurations
    workflow.train_kwargs = train_kwargs
    workflow.train = train
    workflow.CONFIG = CONFIG
    
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args) for variant in args.variants}
        
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
    main()
