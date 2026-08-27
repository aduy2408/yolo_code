#!/usr/bin/env python3
"""End-to-end 100-epoch selectivity runs without surgical transfer or freezing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIGS = {
    "A-R": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_receptance_kvca.yaml",
    "A-P": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_partial_kvca.yaml",
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
    "ranking_summary.json",
)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

    data_yaml = workflow.prepare_fixed_split(
        argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42)
    )
    workflow.validate_split(data_yaml)
    return data_yaml


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ReceptanceKVCompressedAttention, SurgicalPartialKVCompressedAttention

    model = YOLO(CONFIGS[variant])
    model.load(pretrained, smart_transfer=True)
    for parameter in model.model.parameters():
        parameter.requires_grad = True
    layer = model.model.model[16]
    expected = ReceptanceKVCompressedAttention if variant == "A-R" else SurgicalPartialKVCompressedAttention
    if not isinstance(layer, expected):
        raise TypeError(f"{variant}: expected {expected.__name__} at layer 16, got {type(layer).__name__}")
    if model.model.model[-1].stride.tolist() != [4.0]:
        raise ValueError(f"{variant}: expected Detect stride [4.0], got {model.model.model[-1].stride.tolist()}")
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    if len(trainable) != len(list(model.model.parameters())):
        raise AssertionError(f"{variant}: full runner unexpectedly froze parameters")
    return model


def train(variant: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{args.seed}"
    if all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        return run_dir
    seed_everything(args.seed)
    model = model_for(variant, args.pretrained)
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=args.patience, seed=args.seed,
        deterministic=True, amp=args.amp, plots=False, project=str(args.project / variant),
        name=f"seed_{args.seed}", exist_ok=True,
        factorized_tal_target=True, factorized_tal_mode="legacy", factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5, factorized_tal_lambda=0.5, factorized_tal_s_max=32.0,
        factorized_tal_p2_only=True, factorized_tal_warmup_start=0, factorized_tal_warmup_end=0,
    )
    if not all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        raise RuntimeError(f"{variant}: training ended without required artifacts")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ReceptanceKVCompressedAttention

    model = YOLO(run_dir / "weights/best.pt")
    probe = model.model.model[16]
    if isinstance(probe, ReceptanceKVCompressedAttention):
        probe.capture_receptance = True
    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if isinstance(probe, ReceptanceKVCompressedAttention) and probe.last_receptance is not None:
        (run_dir / "receptance_diagnostics.json").write_text(
            json.dumps({"source": "last_validation_or_test_batch", "stats": probe.receptance_statistics()}, indent=2, sort_keys=True) + "\n"
        )
    return metrics


def write_ranking(run_dir: Path, args: argparse.Namespace) -> None:
    from train_levir_scripts import analyze_p2_cbam_ranking as ranking

    images_dir = args.dataset_root / "levir_ship_yolo_seed42/images/test"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.ranking_limit:
        images = images[: args.ranking_limit]
    device = args.device
    if isinstance(device, str) and device.isdigit():
        import torch
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    ranking.EXPECTED_LEVELS["selectivity_full"] = 1
    rows = ranking.inspect_model("selectivity_full", run_dir / "weights/best.pt", images, argparse.Namespace(imgsz=args.imgsz, device=device, expected_seed=args.seed))
    (run_dir / "ranking_summary.json").write_text(json.dumps({"protocol": {"split": "test", "seed": args.seed, "nms_iou": 0.5}, "raw_p2": ranking.descriptive_summary(rows)}, indent=2) + "\n")


def write_manifest(variant: str, run_dir: Path, args: argparse.Namespace) -> None:
    shutil.copy2(CONFIGS[variant], run_dir / "config.yaml")
    manifest = {
        "variant": variant, "seed": args.seed, "split_seed": 42,
        "config": str(CONFIGS[variant]), "pretrained_init": args.pretrained,
        "end_to_end": True, "canonical_transfer": False, "frozen_existing_parameters": False,
        "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
        "batch_size": args.batch_size, "nms_iou": 0.5,
        "ftal": {"mode": "legacy", "tau": 0.75, "kappa": 1.5, "lambda": 0.5, "s_max": 32.0, "p2_only": True, "warmup_start": 0, "warmup_end": 0},
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=list(CONFIGS), default=["A-R"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--project", type=Path, default=ROOT / "runs/levir_kvca_selectivity_full")
    p.add_argument("--pretrained", default="yolov8n.pt")
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--ranking-limit", type=int)
    return p.parse_args(argv)


def main() -> None:
    import train_all_levir_yolov8n_p2_gap_scale_temper as upload_base
    from utils.marimo_ops import require_training_context

    args = parse_args()
    args.data_root, args.dataset_root, args.project = (p.resolve() for p in (args.data_root, args.dataset_root, args.project))
    require_training_context(hf_repo_id=args.hf_repo_id)
    upload_base.REQUIRED = REQUIRED + (("receptance_diagnostics.json",) if "A-R" in args.variants else ())
    uploader = upload_base.Uploader(args.hf_repo_id)
    data_yaml = prepare_split(args)
    for variant in args.variants:
        run_dir = train(variant, data_yaml, args)
        evaluate(run_dir, data_yaml, args)
        write_ranking(run_dir, args)
        write_manifest(variant, run_dir, args)
        required = REQUIRED + (("receptance_diagnostics.json",) if variant == "A-R" else ())
        missing = [path for path in required if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{variant}: missing post-evaluation artifacts: {missing}")
        uploader.upload_run(variant, args.seed, run_dir)


if __name__ == "__main__":
    main()
