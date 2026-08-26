#!/usr/bin/env python3
"""Train the current LEVIR GAP+FTAL reference with independent image augmentations."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_scale_temper as base

ROOT = Path(__file__).resolve().parent
AUGMENTATIONS = {
    "clean_control": {"clean_control_enabled": True, "mosaic": 0.0},
    "random_viewport": {"viewport_enabled": True, "mosaic": 0.0},
    "bbox_occlusion": {"occlusion_enabled": True, "mosaic": 0.0},
    "resolution_degrade": {"resolution_enabled": True, "mosaic": 0.0},
    "viewport_occlusion": {"viewport_enabled": True, "occlusion_enabled": True, "mosaic": 0.0},
    "viewport_mosaic": {"viewport_mosaic_enabled": True, "viewport_enabled": True, "mosaic": 1.0},
    "mosaic_occlusion": {"mosaic_postprocess_enabled": True, "occlusion_enabled": True, "mosaic": 1.0},
    "mosaic_resolution_degrade": {"mosaic_postprocess_enabled": True, "resolution_enabled": True, "mosaic": 1.0},
    "mosaic_occlusion_resolution": {
        "mosaic_postprocess_enabled": True,
        "occlusion_enabled": True,
        "resolution_enabled": True,
        "mosaic": 1.0,
    },
    "viewport_mosaic_occlusion": {
        "viewport_mosaic_enabled": True,
        "viewport_enabled": True,
        "occlusion_enabled": True,
        "mosaic": 1.0,
    },
    "viewport_mosaic_all": {
        "viewport_mosaic_enabled": True,
        "viewport_enabled": True,
        "occlusion_enabled": True,
        "resolution_enabled": True,
        "mosaic": 1.0,
    },
}


def train_variant(augmentation: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    variant = f"gap_factorized_k15_{augmentation}"
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    base.model_for(args.pretrained).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=0, seed=seed, deterministic=True,
        amp=True, plots=False, project=str(args.project / variant),
        name=f"seed_{seed}", exist_ok=True,
        factorized_tal_s_max=32.0, factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15, factorized_tal_p2_only=True,
        factorized_tal_target=True, factorized_tal_mode="legacy", factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5, factorized_tal_lambda=0.5,
        **AUGMENTATIONS[augmentation],
    )
    if not base.training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{variant}: required training artifacts are incomplete")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_gap_factorized_tal_augment")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--augmentations", nargs="+", choices=list(AUGMENTATIONS), default=["clean_control", "random_viewport", "viewport_mosaic"])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    data_yaml = base.prepare_split(args)
    for augmentation in args.augmentations:
        for seed in args.seeds:
            run_dir = train_variant(augmentation, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)


if __name__ == "__main__":
    main()
