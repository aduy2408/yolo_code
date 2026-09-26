#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 plain P2 with Factorized TAL k=1.5."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

# Keep direct execution compatible with the repository-root imports.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_all_levir_yolov8n_p2_gap_factorized_tal as gap
import train_all_levir_yolov8n_p2_gap_scale_temper as base
from utils.marimo_ops import require_training_context


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml"
VARIANT = "plain_p2_gradient_mode_balance"
SETTINGS = {
    "factorized_tal_target": True,
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
    "gradient_mode_balance": True,
    "gradient_mode_count": 2,
    "gradient_mode_iterations": 8,
    "gradient_mode_tiny_size": 32.0,
    "gradient_mode_min_objects": 2,
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


def prepare_split(args: argparse.Namespace) -> Path:
    """Prepare the fixed seed-42 split from the remote checkout layout."""
    path = ROOT / "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py"
    spec = importlib.util.spec_from_file_location("levir_routing", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load split preparation module: {path}")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    split_args = argparse.Namespace(
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        split_seed=args.split_seed,
    )
    data_yaml = workflow.prepare_fixed_split(split_args)
    workflow.validate_split(data_yaml)
    return data_yaml


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    if variant != VARIANT:
        raise ValueError(f"unknown variant: {variant}")
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    base.local_ultralytics()
    from project_ultralytics.training import train_with_loss_adapter

    train_with_loss_adapter(
        model_for(args.pretrained),
        loss_adapter="ftal",
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
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
        "split_seed": args.split_seed,
        "config": CONFIG.name,
        "topology": "Plain P2 -> Detect",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "factorized_tal": SETTINGS,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
        "gradient_mode_balance": True,
        "gradient_mode_count": 2,
        "gradient_mode_iterations": 8,
        "gradient_mode_tiny_size": 32.0,
        "gradient_mode_min_objects": 2,
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
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gradient_mode_balance")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-yaml", default=str(CONFIG))
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-gradient-mode-balance-runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        args.seeds = [args.seed]
    if Path(args.model_yaml).resolve() != CONFIG.resolve():
        raise ValueError(f"Unexpected model YAML: {args.model_yaml}")
    require_training_context(hf_repo_id=args.hf_repo_id)
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = Uploader(args.hf_repo_id)
    data_yaml = prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)
            (run_dir / "factorized_tal_diagnostic.json").write_text(
                json.dumps(
                    {
                        "protocol": "gradient_mode_balance training intervention",
                        "split_seed": 42,
                        "training_seed": seed,
                        "nms_iou": 0.5,
                        "source": "gradient_mode diagnostics are recorded in the training log/runner metrics",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "ranking_summary.json").write_text(
                json.dumps(
                    {
                        "protocol": {"split": "test", "split_seed": 42, "nms_iou": 0.5},
                        "method": "tiny-object mode-balanced P2 localization gradient",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            write_metadata(variant, run_dir, seed, args)
            if not gap.complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
