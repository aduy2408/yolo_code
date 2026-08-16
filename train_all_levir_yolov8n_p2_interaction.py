#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 P2 Semantic-Structural interaction variants (C1-C4)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import train_all_levir_yolov8n_p2_gap_scale_temper as base

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "c1_cross_injection": {},
    "c2_agreement": {},
    "c3_polarity": {},
    "c4_rank4": {},
}
CONFIGS = {
    "c1_cross_injection": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c1_cross_injection.yaml",
    "c2_agreement": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c2_agreement.yaml",
    "c3_polarity": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c3_polarity.yaml",
    "c4_rank4": ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_c4_rank4.yaml",
}

DEFAULT_VARIANTS = ("c1_cross_injection", "c2_agreement", "c3_polarity", "c4_rank4")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
    "interaction_diagnostic.json",
    "ranking_summary.json",
)


def model_for(variant: str, pretrained: str):
    base.local_ultralytics()
    from ultralytics import YOLO

    config_path = CONFIGS[variant]
    model = YOLO(config_path)
    model.load(pretrained, smart_transfer=True)
    return model


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if base.training_complete(run_dir, args.epochs):
        return run_dir
    base.seed_everything(seed)
    model = model_for(variant, args.pretrained)
    model.train(
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
    )
    if not base.training_complete(run_dir, args.epochs):
        raise RuntimeError(f"{variant}: required training artifacts are incomplete")
    return run_dir


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    base.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    config_path = CONFIGS[variant]
    shutil.copy2(config_path, run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "split_seed": 42,
        "config": config_path.name,
        "topology": f"P2 -> {variant} -> shared Detect",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Uploader(base.Uploader):
    def __init__(self, repo_id: str, skip: bool = False) -> None:
        self.skip = skip
        if skip:
            return
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: HF_TOKEN is not set. Uploading will be skipped.")
            self.skip = True
            return
        super().__init__(repo_id)

    def upload_run(self, variant: str, seed: int, run_dir: Path) -> None:
        if self.skip:
            print(f"Skipping upload for {variant} seed {seed}")
            return
        old = base.REQUIRED
        try:
            base.REQUIRED = REQUIRED
            super().upload_run(variant, seed, run_dir)
        finally:
            base.REQUIRED = old


def complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in REQUIRED) and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts.train_all_levir_pathways import prepare_fixed_split as prepare_scene_split
    data_yaml = prepare_scene_split(
        argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42)
    )
    return data_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_interaction")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-interaction-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--ranking-limit", type=int, help="Debug only; full test split when omitted")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to Hugging Face")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    try:
        args.device = int(args.device)
    except ValueError:
        pass
    uploader = Uploader(args.hf_repo_id, skip=args.skip_upload)
    data_yaml = prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            base.evaluate(run_dir, data_yaml, args)
            rows = base.raw_p2_rows(run_dir, args)
            diag = base.diagnose_from_raw(rows)
            (run_dir / "interaction_diagnostic.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            base.ranking_summary(rows, run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            if not complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
