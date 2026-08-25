#!/usr/bin/env python3
"""Train/evaluate/upload clean top-hat + GAP P2 Factorized TAL, without GGCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import train_all_levir_yolov8n_p2_gap_factorized_tal as gap


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_channel_only.yaml"
CONFIG = ROOT / "models_related/models_config/yolov8/levir/generated_input_cues/yolov8n_p2_tophat_gap_factorized_tal.yaml"
VARIANT = "tophat_gap_factorized_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}

_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
_config["backbone"][0] = [-1, 1, "InputCueConv", [64, 3, 2, "top_hat"]]
CONFIG.parent.mkdir(parents=True, exist_ok=True)
CONFIG.write_text(yaml.safe_dump(_config, sort_keys=False), encoding="utf-8")

# Reuse the canonical GAP + FTAL train/eval/upload pipeline with this exact config.
gap.CONFIG = CONFIG
gap.VARIANTS = {VARIANT: SETTINGS}
gap.DEFAULT_VARIANTS = (VARIANT,)


def model_for(pretrained: str):
    gap.base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention, Detect, InputCueConv, copy_rgb_stem_weights
    import torch

    rgb_model = YOLO(pretrained)
    model = YOLO(CONFIG)
    model.load(pretrained, smart_transfer=False)
    stem = model.model.model[0]
    if not isinstance(stem, InputCueConv):
        raise TypeError(f"expected InputCueConv top-hat stem, got {type(stem).__name__}")
    copy_rgb_stem_weights(model.model, rgb_model.model)
    with torch.no_grad():
        stem.conv.weight[:, 3:].zero_()

    layers, head = model.model.model, model.model.model[-1]
    if not any(isinstance(module, ChannelAttention) for module in layers):
        raise TypeError("top-hat GAP FTAL is missing P2 ChannelAttention")
    if not isinstance(head, Detect) or head.f != [19] or head.stride.tolist() != [4.0]:
        raise ValueError(f"expected top-hat GAP P2 Detect from [19], stride [4.0], got {head.f}, {head.stride.tolist()}")
    if bool(getattr(head, "ggcf_refine", False)) or bool(getattr(head, "ggcf_geometry", False)):
        raise TypeError("top-hat GAP FTAL unexpectedly contains GGCF")
    return model


gap.base.model_for = model_for


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_tophat_gap_factorized_tal_noggcf")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-tophat-gap-ftal-noggcf-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = gap.Uploader(args.hf_repo_id)
    data_yaml = gap.base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = gap.train(variant, data_yaml, seed, args)
            gap.base.evaluate(run_dir, data_yaml, args)
            rows = gap.base.raw_p2_rows(run_dir, args)
            diag = gap.base.diagnose_from_raw(rows)
            (run_dir / "factorized_tal_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            gap.base.ranking_summary(rows, run_dir, args)
            gap.write_metadata(variant, run_dir, seed, args)
            if not gap.complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
