#!/usr/bin/env python3
"""Train the matched LEVIR seed-42 P2 evidence-branch ablations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

import train_all_levir_yolov8n_p2_gap_scale_temper as base
from utils.marimo_ops import require_training_context

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
VARIANTS = {
    "extra8_joint": (CONFIG_ROOT / "yolov8n_p2_extra8_joint.yaml", {}),
    "isolated_evidence8": (CONFIG_ROOT / "yolov8n_p2_gradient_isolated_evidence.yaml", {"evidence_aux_gain": 1.0}),
    "scale_disappearance8": (CONFIG_ROOT / "yolov8n_p2_scale_disappearance_evidence.yaml", {}),
    "resolution_extra8": (CONFIG_ROOT / "yolov8n_p2_extra8_joint.yaml", {"resolution_enabled": True, "mosaic_postprocess_enabled": True}),
    "resolution_conditioned8": (CONFIG_ROOT / "yolov8n_p2_resolution_conditioned_evidence.yaml", {"resolution_enabled": True, "mosaic_postprocess_enabled": True, "aug_state_gain": 0.1}),
}


def preflight_variant(variant: str) -> None:
    """Reject a mixed or mislabeled variant before dataset preparation or training."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown evidence variant: {variant}")
    config, settings = VARIANTS[variant]
    if not config.is_file():
        raise FileNotFoundError(config)
    model = yaml.safe_load(config.read_text(encoding="utf-8"))
    layers = model.get("backbone", []) + model.get("head", [])
    modules = [layer[2] for layer in layers]
    expected_module = "ScaleDisappearanceEvidence" if "scale_disappearance" in variant else (
        "AugmentationAwareEvidence" if "conditioned" in variant else "GradientIsolatedEvidence"
    )
    if expected_module not in modules:
        raise ValueError(f"{variant}: config does not contain {expected_module}")
    detect = layers[-1]
    if detect[2] != "Detect":
        raise ValueError(f"{variant}: final layer must be Detect, got {detect[2]}")
    expected_detect_from = [21] if expected_module == "ScaleDisappearanceEvidence" else [20]
    if detect[0] != expected_detect_from:
        raise ValueError(f"{variant}: expected P2-only Detect from {expected_detect_from}, got {detect[0]}")
    if "resolution" in variant and not settings.get("resolution_enabled"):
        raise ValueError(f"{variant}: resolution_enabled must be true")
    if "resolution" in variant and not settings.get("mosaic_postprocess_enabled"):
        raise ValueError(f"{variant}: canonical Mosaic/RandomPerspective path is not pinned")


def model_for(variant: str, pretrained: str):
    base.local_ultralytics()
    from ultralytics import YOLO

    config, _ = VARIANTS[variant]
    model = YOLO(config)
    model.load(pretrained, smart_transfer=True)
    return model


def train_variant(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    model_for(variant, args.pretrained).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=args.patience, seed=seed, deterministic=True,
        amp=True, plots=False, project=str(args.project / variant), name=f"seed_{seed}", exist_ok=True,
        factorized_tal_target=True, factorized_tal_mode="legacy", factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5, factorized_tal_lambda=0.5, factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5, factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True, iou=0.5, **VARIANTS[variant][1],
    )
    if not base.training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{variant}: required training artifacts are incomplete")
    return run_dir


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    shutil.copy2(VARIANTS[variant][0], run_dir / "config.yaml")
    manifest = {
        "variant": variant, "seed": seed, "split_seed": 42,
        "config": VARIANTS[variant][0].name, "epochs": args.epochs, "patience": 0,
        "nms_iou": 0.5, "factorized_tal_mode": "legacy", "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5, "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0, "factorized_tal_p2_only": True,
        "variant_settings": VARIANTS[variant][1],
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_evidence_branches")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    require_training_context(hf_repo_id=args.hf_repo_id)
    if args.epochs != 100:
        raise ValueError("standard evidence ablations require --epochs 100")
    if args.patience != 0:
        raise ValueError("standard evidence ablations require --patience 0")
    if args.seeds != [42]:
        raise ValueError("standard evidence ablations require exactly --seeds 42")
    for variant in args.variants:
        preflight_variant(variant)
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = base.Uploader(args.hf_repo_id)
    data_yaml = base.prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train_variant(variant, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)
            rows = base.raw_p2_rows(run_dir, args)
            (run_dir / "factorized_tal_diagnostic.json").write_text(
                json.dumps(base.diagnose_from_raw(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            base.ranking_summary(rows, run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
