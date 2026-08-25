#!/usr/bin/env python3
"""Train/evaluate/upload clean top-hat + plain P2 Factorized TAL, without GGCF."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

import train_all_levir_yolov8n_p2_plain_factorized_tal as plain


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml"
CONFIG = ROOT / "models_related/models_config/yolov8/levir/generated_input_cues/yolov8n_p2_tophat_plain_factorized_tal.yaml"
VARIANT = "tophat_plain_factorized_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}

_config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
_config["ggcf_refine"] = False
_config["ggcf_geometry"] = False
_config["backbone"][0] = [-1, 1, "InputCueConv", [64, 3, 2, "top_hat"]]
CONFIG.parent.mkdir(parents=True, exist_ok=True)
CONFIG.write_text(yaml.safe_dump(_config, sort_keys=False), encoding="utf-8")

plain.CONFIG = CONFIG
plain.VARIANT = VARIANT
plain.SETTINGS = SETTINGS


def model_for(pretrained: str):
    plain.base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect, InputCueConv, copy_rgb_stem_weights

    rgb_model = YOLO(pretrained)
    model = YOLO(CONFIG)
    model.load(pretrained, smart_transfer=False)
    stem = model.model.model[0]
    if not isinstance(stem, InputCueConv):
        raise TypeError(f"expected InputCueConv top-hat stem, got {type(stem).__name__}")
    copy_rgb_stem_weights(model.model, rgb_model.model)
    with __import__("torch").no_grad():
        stem.conv.weight[:, 3:].zero_()

    layers, head = model.model.model, model.model.model[-1]
    if not isinstance(head, Detect) or head.f != [18] or head.stride.tolist() != [4.0]:
        raise ValueError(f"expected top-hat plain P2 Detect from [18], stride [4.0], got {head.f}, {head.stride.tolist()}")
    if any(type(module).__name__ == "ChannelAttention" for module in layers):
        raise TypeError("top-hat plain FTAL unexpectedly contains GAP ChannelAttention")
    if bool(getattr(head, "ggcf_refine", False)) or bool(getattr(head, "ggcf_geometry", False)):
        raise TypeError("top-hat plain FTAL unexpectedly contains GGCF")
    return model


plain.model_for = model_for


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_tophat_plain_factorized_tal")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-tophat-plain-ftal-noggcf-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = plain.Uploader(args.hf_repo_id)
    data_yaml = plain.base.prepare_split(args)
    for seed in args.seeds:
        run_dir = plain.train(VARIANT, data_yaml, seed, args)
        plain.base.evaluate(run_dir, data_yaml, args)
        rows = plain.base.raw_p2_rows(run_dir, args)
        (run_dir / "factorized_tal_diagnostic.json").write_text(
            json.dumps(plain.base.diagnose_from_raw(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        plain.base.ranking_summary(rows, run_dir, args)
        plain.write_metadata(VARIANT, run_dir, seed, args)
        if not plain.gap.complete(run_dir, args.epochs):
            raise RuntimeError(f"{VARIANT}: required post-evaluation artifacts are incomplete")
        uploader.upload_run(VARIANT, seed, run_dir)


if __name__ == "__main__":
    main()
