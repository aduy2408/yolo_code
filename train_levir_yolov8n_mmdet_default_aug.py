#!/usr/bin/env python3
"""YOLOv8n LevirShip baseline with an explicit MMDetection RTMDet-style augmentation map.

The mapping keeps Ultralytics' geometry compatible with the MMDetection default
RTMDet tiny pipeline: CachedMosaic + RandomResize(0.5, 2.0) + RandomCrop,
YOLOX HSV, RandomFlip(0.5), Pad(114), and CachedMixUp(0.5). Ultralytics does
not expose the cached transforms or an exact crop equivalent, so those two
operations are documented as approximations rather than silently omitted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets/levir_ship_yolo/levir_ship.yaml"
VARIANT = "yolov8n_mmdet_default_aug"
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
)

# RTMDet tiny's default pipeline, expressed using Ultralytics controls where
# there is a faithful or useful equivalent.
MMDET_DEFAULT_AUGMENTATION = {
    "mosaic": 1.0,
    "mixup": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.5,  # RandomResize ratio_range=(0.5, 2.0), approximate
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 0,
    "erasing": 0.0,
    "auto_augment": None,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        fork = ROOT / "models_related/ultralytics"
        sys.path.insert(0, str(fork))
        from ultralytics import YOLO
    return YOLO


def validate_dataset(data_yaml: Path) -> None:
    import yaml
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split in ("train", "val", "test"):
        image_dir = root / payload[split]
        if not image_dir.is_dir() or not any(image_dir.iterdir()):
            raise FileNotFoundError(f"Missing non-empty {split} image directory: {image_dir}")


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / p).is_file() for p in ("weights/best.pt", "weights/last.pt", "args.yaml")) and results.is_file() and results.stat().st_size > 0


def train(args: argparse.Namespace, run_dir: Path) -> None:
    if training_complete(run_dir):
        print(f"Reusing completed training: {run_dir}", flush=True)
        return
    seed_everything(args.seed)
    YOLO = load_yolo()
    model = YOLO(args.pretrained)
    model.train(
        data=str(args.data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch_size, device=args.device, workers=args.workers,
        patience=args.patience, seed=args.seed, deterministic=True, amp=args.amp,
        plots=False, project=str(args.project), name=VARIANT, exist_ok=True,
        **MMDET_DEFAULT_AUGMENTATION,
    )
    if not training_complete(run_dir):
        raise RuntimeError(f"Training did not produce required artifacts: {run_dir}")


def evaluate(args: argparse.Namespace, run_dir: Path) -> dict[str, float | str]:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    YOLO = load_yolo()
    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(args.data_yaml), split=split, imgsz=args.imgsz,
            batch=args.batch_size, device=args.device, workers=args.workers,
            plots=False, iou=0.5, project=str(run_dir / "evaluation"),
            name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_metadata(args: argparse.Namespace, run_dir: Path) -> None:
    shutil.copy2(args.data_yaml, run_dir / "config.yaml")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    manifest = {
        "variant": VARIANT, "pretrained": args.pretrained, "seed": args.seed,
        "git_sha": sha, "data_yaml": str(args.data_yaml),
        "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
        "batch_size": args.batch_size, "nms_iou": 0.5,
        "augmentation": MMDET_DEFAULT_AUGMENTATION,
        "mmdetection_reference": "rtmdet_tiny_8xb32-300e_coco.py",
        "known_approximations": ["CachedMosaic is mapped to mosaic=1.0", "RandomCrop is approximated by YOLO resize/letterbox"],
        "command": sys.argv,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload(args: argparse.Namespace, run_dir: Path) -> None:
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    from huggingface_hub import HfApi
    missing = [p for p in REQUIRED if not (run_dir / p).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload: {missing}")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", exist_ok=True)
    remote = f"runs/{VARIANT}/seed_{args.seed}"
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset")
    expected = {f"{remote}/{p}" for p in REQUIRED}
    remote_files = set(api.list_repo_files(args.hf_repo_id, repo_type="dataset"))
    missing_remote = sorted(expected - remote_files)
    if missing_remote:
        raise RuntimeError(f"Remote upload verification failed: {missing_remote}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/{marker.name}", repo_id=args.hf_repo_id, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_mmdet_default_aug")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-ship-yolov8n-mmdet-default-aug")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_yaml = args.data_yaml.resolve()
    args.project = args.project.resolve()
    validate_dataset(args.data_yaml)
    run_dir = args.project / VARIANT / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train(args, run_dir)
    evaluate(args, run_dir)
    write_metadata(args, run_dir)
    if args.upload:
        upload(args, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "variant": VARIANT, "nms_iou": 0.5, "uploaded": args.upload}, sort_keys=True))


if __name__ == "__main__":
    main()
