#!/usr/bin/env python3
"""Run the eight YOLOv9t-P2 LEVIR-Ship ablations through the shared workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
from train_levir_scripts.yolov9t_p2_run_matrix import RUNS

workflow.EXPERIMENT = "levir_yolov9t_p2"
workflow.HF_REPO = "duyle2408/levir-yolov9t-p2-seed42"
workflow.VARIANTS = {key: spec["config"] for key, spec in RUNS.items()}

_BASE_TRAIN_KWARGS = workflow.train_kwargs
_BASE_MODEL_FOR = workflow.model_for


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    """Merge the shared protocol with the selected run's FTAL settings."""
    kwargs = _BASE_TRAIN_KWARGS(args, data_yaml, seed, amp)
    variant = getattr(args, "current_variant", None)
    if variant is None:
        raise ValueError("current_variant must be set before building train kwargs")
    kwargs.update(RUNS[variant]["train_kwargs"])
    return kwargs


def model_for(variant: str, pretrained: str):
    """Build a YOLOv9t-P2 model and verify its four-level Detect head."""
    model = _BASE_MODEL_FOR(variant, pretrained)
    head = model.model.model[-1]
    if getattr(head, "nl", None) != 4:
        raise ValueError(f"{variant}: expected four-level P2/P3/P4/P5 Detect head")
    strides = [float(value) for value in head.stride]
    if strides != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"{variant}: unexpected Detect strides {strides}")
    return model


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    args.current_variant = variant
    return _BASE_TRAIN(variant, seed, data_yaml, amp, args)


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace, amp: bool = True) -> bool:
    args.current_variant = variant
    return _BASE_SMOKE(variant, data_yaml, args, amp)


_BASE_TRAIN = workflow.train
_BASE_SMOKE = workflow.smoke


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(workflow.VARIANTS), default=list(workflow.VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{workflow.EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov9t.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=workflow.HF_REPO)
    return parser.parse_args(argv)


def main() -> None:
    workflow.prepare_fixed_split = workflow.prepare_fixed_split
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.train = train
    workflow.smoke = smoke
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
