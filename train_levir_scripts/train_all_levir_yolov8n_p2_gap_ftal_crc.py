#!/usr/bin/env python3
"""Train/evaluate the narrow GAP+FTAL P2 classification ring-context test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_ftal_crc"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-ftal-crc"
workflow.VARIANTS = {
    "plain_ftal_crc": CONFIG_ROOT / "yolov8n_p2_fpn_only_plain_crc.yaml",
    "gap_ftal_crc": CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_ring_context_cls.yaml",
    "gap_ftal_crc_gate_only": CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_ring_context_cls.yaml",
    "gap_ftal_crc_contrast_only": CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_ring_context_cls.yaml",
}

_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for
_base_smoke = workflow.smoke
_base_train = workflow.train


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
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


def model_for(variant: str, pretrained: str):
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import ChannelAttention, Detect, RingContextCls

    layers = model.model.model
    head = layers[-1]
    if variant in {"gap_ftal_crc", "gap_ftal_crc_gate_only", "gap_ftal_crc_contrast_only"}:
        if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
            raise ValueError(f"{variant}: expected P2 -> ChannelAttention(avg) -> Detect([19])")
    else:
        if not isinstance(head, Detect) or head.f != [18]:
            raise ValueError(f"{variant}: expected P2 -> Detect([18])")
    if head.stride.tolist() != [4.0] or head.nl != 1:
        raise ValueError(f"{variant}: expected P2-only Detect stride [4], got {head.stride.tolist()}")
    if not isinstance(head.cls_ring_context[0], RingContextCls) or not head.ring_context or head.ring_radius != 5:
        raise ValueError(f"{variant}: ring context did not resolve as R5 P2 cls-only adapter")
    return model


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    original_train_kwargs = workflow.train_kwargs
    def custom_train_kwargs(*args_kwargs, **kwargs_kwargs):
        kwargs = original_train_kwargs(*args_kwargs, **kwargs_kwargs)
        if variant == "gap_ftal_crc_gate_only":
            kwargs.update(crc_gate_coeff=0.5, crc_contrast_coeff=0.0)
        elif variant == "gap_ftal_crc_contrast_only":
            kwargs.update(crc_gate_coeff=0.0, crc_contrast_coeff=0.2)
        else:
            kwargs.update(crc_gate_coeff=0.5, crc_contrast_coeff=0.2)
        return kwargs
    workflow.train_kwargs = custom_train_kwargs
    try:
        return _base_smoke(variant, data_yaml, args, amp)
    finally:
        workflow.train_kwargs = original_train_kwargs


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    original_train_kwargs = workflow.train_kwargs
    def custom_train_kwargs(*args_kwargs, **kwargs_kwargs):
        kwargs = original_train_kwargs(*args_kwargs, **kwargs_kwargs)
        if variant == "gap_ftal_crc_gate_only":
            kwargs.update(crc_gate_coeff=0.5, crc_contrast_coeff=0.0)
        elif variant == "gap_ftal_crc_contrast_only":
            kwargs.update(crc_gate_coeff=0.0, crc_contrast_coeff=0.2)
        else:
            kwargs.update(crc_gate_coeff=0.5, crc_contrast_coeff=0.2)
        return kwargs
    workflow.train_kwargs = custom_train_kwargs
    try:
        return _base_train(variant, seed, data_yaml, amp, args)
    finally:
        workflow.train_kwargs = original_train_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(workflow.VARIANTS), default=list(workflow.VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{workflow.EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=workflow.HF_REPO)
    return parser.parse_args()


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.smoke = smoke
    workflow.train = train
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
