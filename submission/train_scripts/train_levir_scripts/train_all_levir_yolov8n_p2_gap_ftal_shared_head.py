#!/usr/bin/env python3
"""Train/evaluate GAP+FTAL Detect head sharing topology variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"

workflow.EXPERIMENT = "levir_yolov8n_p2_gap_ftal_shared_head"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-gap-ftal-shared-head-seed42"
workflow.VARIANTS = {
    "gap_ftal_decoupled": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
    "gap_ftal_share1": CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_head_share1.yaml",
    "gap_ftal_fully_shared": CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_head_fully_shared.yaml",
}
EXPECTED_MODE = {
    "gap_ftal_decoupled": "none",
    "gap_ftal_share1": "share1",
    "gap_ftal_fully_shared": "full",
}

_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for
_base_evaluate = workflow.evaluate


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
    from ultralytics.nn.modules import ChannelAttention, Detect

    layers = model.model.model
    head = layers[-1]
    if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
        raise ValueError(f"{variant}: expected P2 -> ChannelAttention(avg) -> Detect([19])")
    if head.stride.tolist() != [4.0] or head.nl != 1:
        raise ValueError(f"{variant}: expected P2-only Detect stride [4], got {head.stride.tolist()}")
    if head.head_share_mode != EXPECTED_MODE[variant]:
        raise ValueError(f"{variant}: expected head_share_mode={EXPECTED_MODE[variant]}, got {head.head_share_mode}")
    if head.ring_context or getattr(head, "quality_head", False) or getattr(head, "cls_geometry_fuse", False):
        raise ValueError(f"{variant}: expected topology-only Detect head sharing")
    return model


def _model_stats(model) -> dict[str, float]:
    params = sum(p.numel() for p in model.model.parameters())
    try:
        gf = float(model.model.info(verbose=False, imgsz=512)[-1])
    except Exception:
        gf = -1.0
    return {"params": params, "gflops_512": gf}


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    metrics = _base_evaluate(run_dir, data_yaml, args)
    if "params" not in metrics:
        variant = run_dir.parent.name
        model = model_for(variant, str(run_dir / "weights/best.pt"))
        metrics.update(_model_stats(model), nms_iou=0.5)
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    return args


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.evaluate = evaluate
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
