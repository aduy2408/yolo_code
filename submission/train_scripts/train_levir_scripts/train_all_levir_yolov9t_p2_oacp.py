#!/usr/bin/env python3
"""Run one upload-required YOLOv9t P2-only + OACP LEVIR-Ship experiment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

CONFIG = ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_no_p5_levir.yaml"
VARIANT = "yolov9t_p2_no_p5_oacp"
workflow.EXPERIMENT = "levir_yolov9t_p2_no_p5_oacp"
workflow.HF_REPO = "duyle2408/levir-yolov9t-p2-no-p5-oacp-seed42"
workflow.VARIANTS = {VARIANT: CONFIG}

# The shared Ultralytics data pipeline reads this before constructing train transforms.
os.environ["YOLO_CONTEXT_AUG"] = "oacp"

_BASE_MODEL_FOR = workflow.model_for
_BASE_TRAIN = workflow.train
_BASE_SMOKE = workflow.smoke
_BASE_EVALUATE = workflow.evaluate


def model_for(variant: str, pretrained: str):
    model = _BASE_MODEL_FOR(variant, pretrained)
    head = model.model.model[-1]
    if getattr(head, "nl", None) != 3:
        raise ValueError(f"{variant}: expected three-level P2/P3/P4 Detect head")
    strides = [float(value) for value in head.stride]
    if strides != [4.0, 8.0, 16.0]:
        raise ValueError(f"{variant}: unexpected Detect strides {strides}")
    return model


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args):
    args.current_variant = variant
    return _BASE_TRAIN(variant, seed, data_yaml, amp, args)


def smoke(variant: str, data_yaml: Path, args, amp: bool = True):
    args.current_variant = variant
    return _BASE_SMOKE(variant, data_yaml, args, amp)


def evaluate(run_dir: Path, data_yaml: Path, args):
    metrics = _BASE_EVALUATE(run_dir, data_yaml, args)
    manifest = {
        "experiment": workflow.EXPERIMENT,
        "variant": VARIANT,
        "config": str(CONFIG),
        "architecture": "YOLOv9t P2-only, no P5 Detect output",
        "augmentation": "oacp",
        "commit_sha": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "seed": args.seeds[0],
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "hf_repo_id": args.hf_repo_id,
        "metrics": metrics,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return metrics


def parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / f"runs/{workflow.EXPERIMENT}")
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
    workflow.model_for = model_for
    workflow.train = train
    workflow.smoke = smoke
    workflow.evaluate = evaluate
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
