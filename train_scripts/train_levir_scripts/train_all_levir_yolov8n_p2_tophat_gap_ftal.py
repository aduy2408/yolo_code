#!/usr/bin/env python3
"""Run one LEVIR YOLOv8n-P2 experiment combining top-hat input cues with GAP+FTAL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_gap_ftal_ggcf as gap_runner


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
GAP_CONFIG = CONFIG_ROOT / "yolov8n_p2_fpn_only_gap_ggcf.yaml"
COMBINED_CONFIG = CONFIG_ROOT / "generated_input_cues/yolov8n_p2_tophat_gap_ftal.yaml"

COMBINED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
_config = yaml.safe_load(GAP_CONFIG.read_text(encoding="utf-8"))
_config["backbone"][0] = [-1, 1, "InputCueConv", [64, 3, 2, "top_hat"]]
COMBINED_CONFIG.write_text(yaml.safe_dump(_config, sort_keys=False), encoding="utf-8")

workflow = gap_runner.workflow
workflow.EXPERIMENT = "levir_yolov8n_p2_tophat_gap_ftal"
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-tophat-gap-ftal"
workflow.VARIANTS = {"top_hat_gap_ftal": COMBINED_CONFIG}


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = gap_runner.train_kwargs(args, data_yaml, seed, amp)
    kwargs["ggcf_assign_refined"] = True
    kwargs["ggcf_tal_diagnostics"] = False
    return kwargs


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention, Detect, InputCueConv, copy_rgb_stem_weights

    rgb_model = YOLO(pretrained)
    model = YOLO(COMBINED_CONFIG)
    model.load(pretrained, smart_transfer=False)

    stem = model.model.model[0]
    if not isinstance(stem, InputCueConv):
        raise TypeError(f"Expected InputCueConv stem, got {type(stem).__name__}")
    copy_rgb_stem_weights(model.model, rgb_model.model)
    with __import__("torch").no_grad():
        stem.conv.weight[:, 3:].zero_()

    layers = model.model.model
    head = layers[-1]
    if not isinstance(layers[19], ChannelAttention) or not isinstance(head, Detect) or head.f != [19]:
        raise ValueError("Expected P2 -> ChannelAttention -> Detect([19])")
    if head.stride.tolist() != [4.0] or head.nl != 1 or not head.ggcf_refine:
        raise ValueError(f"Expected P2-only GGCF Detect stride [4], got {head.stride.tolist()}")
    head.ggcf_geometry = True
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=workflow.HF_REPO)
    parser.set_defaults(variants=["top_hat_gap_ftal"])
    parser.set_defaults(runner=Path(__file__).resolve())
    return parser.parse_args(argv)


def main() -> None:
    workflow.train_kwargs = train_kwargs
    workflow.model_for = model_for
    workflow.Uploader = gap_runner.Uploader
    workflow.parse_args = parse_args
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {"top_hat_gap_ftal": True}
    if not args.no_smoke:
        args._variant = "top_hat_gap_ftal"
        amp["top_hat_gap_ftal"] = workflow.smoke("top_hat_gap_ftal", data_yaml, args)
    if args.smoke_only:
        return
    for seed in args.seeds:
        args._variant = "top_hat_gap_ftal"
        run_dir = workflow.train("top_hat_gap_ftal", seed, data_yaml, amp["top_hat_gap_ftal"], args)
        workflow.evaluate(run_dir, data_yaml, args)
        workflow.write_summaries(args)
        if uploader:
            uploader.upload_run(run_dir, "top_hat_gap_ftal", seed)
            uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
