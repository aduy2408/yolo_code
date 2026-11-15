#!/usr/bin/env python3
"""Train, evaluate, summarize, and upload the YOLOv8n early P2 representation sweep."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "levir_yolov8n_p2_representation_sweep"
HF_REPO = "duyle2408/levir-yolov8n-p2-representation-sweep-3seed"
VARIANTS = {
    "plain_control": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml",
    "b_deep_supervision": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_b_deep_supervision.yaml",
    "canonical_teacher": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_canonical_teacher.yaml",
    "raw_sidecar_supervised": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_raw_sidecar_supervised.yaml",
    "raw_sidecar_control": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_raw_sidecar_supervised.yaml", # Control (D0) using same config, raw_sidecar_gain=0
}

workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS
_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for
_base_smoke = workflow.smoke
_base_train = workflow.train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
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


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "_active_variant", "")
    
    # Configure hyperparams based on variant
    if variant == "b_deep_supervision":
        kwargs.update(p2_deep_sup_gain=1.0)
    elif variant == "canonical_teacher":
        kwargs.update(canonical_teacher_gain=1.0)
    elif variant == "raw_sidecar_supervised":
        kwargs.update(raw_sidecar_gain=1.0)
    elif variant == "raw_sidecar_control":
        kwargs.update(raw_sidecar_gain=0.0) # Control branch (D0) has no auxiliary supervision
        
    return kwargs


def model_for(variant: str, pretrained: str):
    return _base_model_for(variant, pretrained)


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    args._active_variant = variant
    return _base_smoke(variant, data_yaml, args, amp)


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    args._active_variant = variant
    return _base_train(variant, seed, data_yaml, amp, args)


def main() -> None:
    args = parse_args()
    args.variants = list(VARIANTS)
    args.runner = Path(__file__).resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.smoke = smoke
    workflow.train = train
    
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    
    amp = {variant: args.amp for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: smoke(variant, data_yaml, args, amp=args.amp) for variant in args.variants}
        
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
