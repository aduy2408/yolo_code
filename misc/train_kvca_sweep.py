#!/usr/bin/env python3
"""Train, test, and sanity-check the YOLOv8 A_sc KVCA compression sweep."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = REPO_ROOT / "models_related" / "ultralytics"

MODEL_SCALE = "n"
EPOCHS = 200
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = "cuda"
PATIENCE = 30

DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
DATASET_OUT_DIR = REPO_ROOT / "datasets" / "varroa_yolo"
DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"

PROJECT = REPO_ROOT / "runs" / "detect" / "yolo_related" / "runs" / "train_kvca_sweep"
TEST_PROJECT = REPO_ROOT / "runs" / "detect" / "yolo_related" / "runs" / "test_kvca_sweep"
GENERATED_CONFIG_DIR = Path("/tmp") / "varroa_kvca_sweep_scaled_configs"

SWEEP_CONFIGS = [
    REPO_ROOT / "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_avg_p2sr4_p3sr4.yaml",
    REPO_ROOT / "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_avg_dwk_p2sr4_p3sr4.yaml",
    REPO_ROOT / "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_dw_stride_p2sr4_p3sr4.yaml",
    REPO_ROOT / "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_group_weight_p2sr4_p3sr4.yaml",
    REPO_ROOT / "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_avg_dwk_p2sr8_p3sr4.yaml",
]


def prefer_local_ultralytics() -> None:
    """Force imports to use the repo's local Ultralytics fork."""
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


prefer_local_ultralytics()

from ultralytics import YOLO  # noqa: E402


def prepare_data() -> None:
    """Prepare the Varroa YOLO dataset."""
    from misc.prepare_dataset import prepare_dataset

    print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
    prepare_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))


def config_for_scale(config: Path, model_scale: str) -> Path:
    """Copy a config and inject the selected model scale."""
    text = config.read_text()
    out_lines = []
    inserted = False

    for line in text.splitlines():
        if line.startswith("scale:"):
            out_lines.append(f"scale: {model_scale}")
            inserted = True
            continue

        out_lines.append(line)
        if not inserted and line.startswith("nc:"):
            out_lines.append(f"scale: {model_scale}")
            inserted = True

    if not inserted:
        out_lines.insert(0, f"scale: {model_scale}")

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    scaled_config = GENERATED_CONFIG_DIR / f"{config.stem}_{model_scale}{config.suffix}"
    scaled_config.write_text("\n".join(out_lines) + "\n")
    return scaled_config


def run_name_for(config: Path, model_scale: str) -> str:
    """Return the Ultralytics run name for a sweep config."""
    return f"{config.stem}_{model_scale}"


def sanity_check(configs: list[Path], model_scale: str, imgsz: int, device: str) -> list[tuple[str, int]]:
    """Instantiate each config and run one dummy forward pass."""
    results = []
    forward_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    for config in configs:
        model_yaml = config_for_scale(config, model_scale)
        model = YOLO(str(model_yaml))
        model.model.to(forward_device).eval()
        params = sum(p.numel() for p in model.model.parameters())

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=forward_device)
            model.model(dummy)

        print(f"OK {config.name}: params={params} forward_device={forward_device}")
        results.append((config.stem, params))

    return results


def train_config(config: Path, model_scale: str, args: argparse.Namespace) -> None:
    """Train one sweep config."""
    model_yaml = config_for_scale(config, model_scale)
    run_name = run_name_for(config, model_scale)

    print("\n" + "=" * 60)
    print(f"Training: {run_name}")
    print(f"Config:   {model_yaml}")
    print("=" * 60)

    model = YOLO(str(model_yaml))
    model.load(f"yolov8{model_scale}.pt", smart_transfer=True)
    model.train(
        data=str(DATA_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        project=str(PROJECT),
        name=run_name,
    )


def find_best_weights(configs: list[Path], model_scale: str) -> list[Path]:
    """Find best.pt files for this script's run names."""
    run_names = {run_name_for(config, model_scale) for config in configs}
    best_paths = []
    for best_path in PROJECT.glob("*/weights/best.pt"):
        if best_path.parents[1].name in run_names:
            best_paths.append(best_path)
    return sorted(best_paths)


def run_test_inference(configs: list[Path], model_scale: str, args: argparse.Namespace) -> Path | None:
    """Evaluate trained checkpoints on the test split and save a summary CSV."""
    best_paths = find_best_weights(configs, model_scale)
    print(f"\nFound {len(best_paths)} trained checkpoints for testing")
    if not best_paths:
        return None

    rows = []
    for best_path in best_paths:
        run_name = best_path.parents[1].name

        print("\n" + "=" * 60)
        print(f"Testing: {run_name}")
        print(f"Weight:  {best_path}")
        print("=" * 60)

        model = YOLO(str(best_path))
        metrics = model.val(
            data=str(DATA_PATH),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.5,
            project=str(TEST_PROJECT),
            name=run_name,
            exist_ok=True,
        )

        row = (
            run_name,
            float(metrics.box.map50),
            float(metrics.box.map),
            float(metrics.box.mp),
            float(metrics.box.mr),
        )
        rows.append(row)
        print(f"mAP50={row[1]:.4f} | mAP50-95={row[2]:.4f} | P={row[3]:.4f} | R={row[4]:.4f}")

    rows.sort(key=lambda x: x[1], reverse=True)
    TEST_PROJECT.mkdir(parents=True, exist_ok=True)
    summary_path = TEST_PROJECT / "kvca_sweep_test_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(rows)

    print(f"\nSaved test summary: {summary_path}")
    return TEST_PROJECT


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only load configs and run one dummy forward pass.")
    return parser.parse_args()


def main() -> None:
    """Run the KVCA sweep."""
    args = parse_args()

    missing = [config for config in SWEEP_CONFIGS if not config.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing sweep configs: {missing}")

    if GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(GENERATED_CONFIG_DIR)

    sanity_check(SWEEP_CONFIGS, args.scale, args.imgsz, args.device)
    if args.check_only:
        return

    if not args.skip_prepare_data:
        prepare_data()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

    for config in SWEEP_CONFIGS:
        train_config(config, args.scale, args)

    run_test_inference(SWEEP_CONFIGS, args.scale, args)


if __name__ == "__main__":
    main()
