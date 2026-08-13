#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 plain P2 with Factorized TAL k=1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_factorized_tal as gap
import train_all_levir_yolov8n_p2_gap_scale_temper as base


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml"
VARIANT = "plain_p2_factorized_k15"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
}


def model_for(pretrained: str):
    base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention, Detect

    model = YOLO(CONFIG)
    model.load(pretrained, smart_transfer=True)
    layers, head = model.model.model, model.model.model[-1]
    if not isinstance(head, Detect) or head.f != [18] or head.stride.tolist() != [4.0]:
        raise ValueError(f"expected plain P2 Detect from [18], stride [4.0], got {head.f}, {head.stride.tolist()}")
    if any(isinstance(module, ChannelAttention) for module in layers):
        raise TypeError("plain P2 config unexpectedly contains ChannelAttention")
    return model


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    if variant != VARIANT:
        raise ValueError(f"unknown variant: {variant}")
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    model_for(args.pretrained).train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=0,
        seed=seed,
        deterministic=True,
        amp=True,
        plots=False,
        project=str(args.project / variant),
        name=f"seed_{seed}",
        exist_ok=True,
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
        **SETTINGS,
    )
    if not base.training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{variant}: required training artifacts are incomplete")
    return run_dir


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    import shutil

    base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(CONFIG, run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "split_seed": 42,
        "config": CONFIG.name,
        "topology": "Plain P2 -> Detect",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "factorized_tal": SETTINGS,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Uploader(gap.Uploader):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_plain_factorized_tal")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-plain-factorized-tal-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = Uploader(args.hf_repo_id)
    data_yaml = base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)
            rows = base.raw_p2_rows(run_dir, args)
            (run_dir / "factorized_tal_diagnostic.json").write_text(
                json.dumps(base.diagnose_from_raw(rows), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            base.ranking_summary(rows, run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            if not gap.complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
