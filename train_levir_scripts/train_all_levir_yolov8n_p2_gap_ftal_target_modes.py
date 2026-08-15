#!/usr/bin/env python3
"""Train/evaluate GAP + FTAL target-mode tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_ftal_target_modes"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-ftal-target-modes-seed42"
workflow.VARIANTS = {
    "gap_ftal_mass_preserve_l05": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
    "gap_ftal_mass_preserve_l1": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
    "gap_ftal_geometry_l05": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
    "gap_ftal_agreement_gate_l05": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
}

_BASE_TRAIN_KWARGS = workflow.train_kwargs
_BASE_MODEL_FOR = workflow.model_for
_BASE_SMOKE = workflow.smoke

MODES = {
    "gap_ftal_mass_preserve_l05": ("mass_preserve", 0.5),
    "gap_ftal_mass_preserve_l1": ("mass_preserve", 1.0),
    "gap_ftal_geometry_l05": ("geometry", 0.5),
    "gap_ftal_agreement_gate_l05": ("agreement_gate", 0.5),
}


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _BASE_TRAIN_KWARGS(args, data_yaml, seed, amp)
    mode, lam = MODES[args.current_variant]
    kwargs.update(
        factorized_tal_target=True,
        factorized_tal_mode=mode,
        factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5,
        factorized_tal_lambda=lam,
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
    )
    return kwargs


def model_for(variant: str, pretrained: str):
    model = _BASE_MODEL_FOR(variant, pretrained)
    from ultralytics.nn.modules import ChannelAttention, Detect

    layers = model.model.model
    head = layers[-1]
    if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
        raise ValueError(f"{variant}: expected P2 -> GAP ChannelAttention -> Detect([19])")
    if head.stride.tolist() != [4.0] or head.nl != 1:
        raise ValueError(f"{variant}: expected P2-only Detect stride [4], got {head.stride.tolist()}")
    return model


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    args.current_variant = variant
    return _BASE_TRAIN(variant, seed, data_yaml, amp, args)


_BASE_TRAIN = workflow.train


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    args.current_variant = variant
    return _BASE_SMOKE(variant, data_yaml, args, amp)


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
    parser.set_defaults(current_variant="gap_ftal_mass_preserve_l1", runner=Path(__file__).resolve())
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
