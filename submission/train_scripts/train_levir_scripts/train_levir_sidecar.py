#!/usr/bin/env python3
"""Train/evaluate LEVIR P2 sidecar residual fusion architectures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

EXPERIMENT = "levir_yolov8n_p2_sidecar"
HF_REPO = "duyle2408/levir-yolov8n-p2-sidecar"
PLAIN_YAML = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml"
VARIANTS = {
    "linear_residual": CONFIG_ROOT / "yolov8n_p2_sidecar_linear.yaml",
    "joint_residual": CONFIG_ROOT / "yolov8n_p2_sidecar_joint.yaml",
    "norm_joint_residual": CONFIG_ROOT / "yolov8n_p2_sidecar_norm_joint.yaml",
    "replace_joint": CONFIG_ROOT / "yolov8n_p2_sidecar_replace.yaml",
}

workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS
_base_train_kwargs = workflow.train_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
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


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    return _base_train_kwargs(args, data_yaml, seed, amp)


def _copy_semantic_plain_initialization(target, plain) -> int:
    """Copy main backbone components and downstream FPN parameters from plain YOLO P2 initialization."""
    src = plain.model.model
    dst = target.model.model
    stem = dst[0]
    copied = 0

    # Target layer 0 contains standard main cv1/cv2/c2f stem components as its main path.
    stem.main_cv1.load_state_dict(src[0].state_dict(), strict=True)
    stem.main_cv2.load_state_dict(src[1].state_dict(), strict=True)
    stem.main_c2f.load_state_dict(src[2].state_dict(), strict=True)
    copied += sum(len(src[i].state_dict()) for i in (0, 1, 2))

    # Every downstream layer has the same role as plain layer target_index + 2.
    for target_index in range(1, len(dst)):
        source_index = target_index + 2
        source_state = src[source_index].state_dict()
        target_state = dst[target_index].state_dict()
        if list(source_state) != list(target_state):
            raise RuntimeError(
                f"semantic transfer key mismatch: plain layer {source_index} -> target layer {target_index}"
            )
        for key in source_state:
            if source_state[key].shape != target_state[key].shape:
                raise RuntimeError(
                    f"semantic transfer shape mismatch at plain {source_index}/{key}: "
                    f"{source_state[key].shape} vs {target_state[key].shape}"
                )
        dst[target_index].load_state_dict(source_state, strict=True)
        copied += len(source_state)

    return copied


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO

    # First reproduce the exact initialization path used by the established plain P2 baseline.
    plain = YOLO(PLAIN_YAML)
    plain.load(pretrained, smart_transfer=True)

    # Build the new graph from scratch, then copy only semantically corresponding baseline states.
    model = YOLO(VARIANTS[variant])
    copied = _copy_semantic_plain_initialization(model, plain)
    print(f"Copied {copied} parameters from plain P2 initialization")
    return model


def main() -> None:
    args = parse_args()
    workflow.train_kwargs = lambda a, dy, s, amp: train_kwargs(a, dy, s, amp)
    workflow.model_for = model_for
    
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
