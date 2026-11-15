#!/usr/bin/env python3
"""Train/evaluate GAP + Factorized TAL k=1.5 with lambda=1.0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_factorized_tal_lambda1"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-factorized-tal-lambda1"
workflow.VARIANTS = {
    "gap_factorized_k15_lambda1": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
}

_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for
_base_parse_args = workflow.parse_args


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        factorized_tal_target=True,
        factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5,
        factorized_tal_lambda=1.0,
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
    )
    return kwargs


def model_for(variant: str, pretrained: str):
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import ChannelAttention, Detect

    layers = model.model.model
    head = layers[-1]
    if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
        raise ValueError(f"{variant}: expected P2 -> GAP ChannelAttention -> Detect([19])")
    if head.stride.tolist() != [4.0] or head.nl != 1:
        raise ValueError(f"{variant}: expected P2-only Detect stride [4], got {head.stride.tolist()}")
    return model


def parse_args() -> argparse.Namespace:
    parser = _base_parse_args()
    parser.variants = ["gap_factorized_k15_lambda1"]
    parser.seeds = [42]
    parser.project = ROOT.parent / f"runs/{workflow.EXPERIMENT}"
    parser.hf_repo_id = workflow.HF_REPO
    parser.runner = Path(__file__).resolve()
    return parser


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
