#!/usr/bin/env python3
"""Train and test the two TPH/KVCA Varroa configs."""

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "models_related" / "ultralytics"

# ==========================================
# EDIT CONFIG HERE
# ==========================================
MODEL_SCALE = "n"  # n, s, m, l, x
EPOCHS = 200
IMGSZ = 640
BATCH = 16
WORKERS = 4
DEVICE = "cuda"
PATIENCE = 30

RUN_PREPARE_DATA = True
DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"

PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "train"
TEST_PROJECT = ROOT / "runs" / "detect" / "yolo_related" / "runs" / "test"
GENERATED_CONFIG_DIR = Path("/tmp") / "varroa_train_all_2_configs"

CONFIGS_TO_RUN = [
    ROOT / "models_related" / "models_config" / "yolov8" / "yolov8_varroa_tph_kvencoder_repc2f_ensimam_p2.yaml",
    ROOT / "models_related" / "models_config" / "yolov5" / "yolov5-tph-kvca-cbam-p2.yaml",
]
# ==========================================


def prefer_local_ultralytics() -> None:
    """Force imports to use the repo's local Ultralytics fork."""
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


prefer_local_ultralytics()

from ultralytics import YOLO


def prepare_data() -> None:
    """Prepare the Varroa YOLO dataset."""
    from misc.prepare_dataset import prepare_dataset

    print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
    prepare_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))


def config_for_scale(config: Path) -> Path:
    """Copy a config and inject the selected model scale."""
    text = config.read_text()
    out_lines = []
    inserted = False

    for line in text.splitlines():
        if line.startswith("scale:"):
            out_lines.append(f"scale: {MODEL_SCALE}")
            inserted = True
            continue

        out_lines.append(line)
        if not inserted and line.startswith("nc:"):
            out_lines.append(f"scale: {MODEL_SCALE}")
            inserted = True

    if not inserted:
        out_lines.insert(0, f"scale: {MODEL_SCALE}")

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    scaled_config = GENERATED_CONFIG_DIR / f"{config.stem}_{MODEL_SCALE}{config.suffix}"
    scaled_config.write_text("\n".join(out_lines) + "\n")
    return scaled_config


def pretrained_for(config: Path) -> str:
    """Return matching pretrained weights for a config family."""
    return f"yolov5{MODEL_SCALE}.pt" if "yolov5" in config.parts else f"yolov8{MODEL_SCALE}.pt"


def train_config(config: Path) -> None:
    """Train one config."""
    if not config.is_file():
        raise FileNotFoundError(f"Missing config: {config}")

    model_yaml = config_for_scale(config)
    run_name = f"{config.stem}_{MODEL_SCALE}"

    print("\n" + "=" * 60)
    print(f"Training: {run_name}")
    print(f"Config:   {model_yaml}")
    print("=" * 60)

    model = YOLO(str(model_yaml))
    model.load(pretrained_for(config), smart_transfer=True)

    model.train(
        data=str(DATA_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        patience=PATIENCE,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        # EXPERIMENTAL: boundary-aware contrastive localization ablation.
        # boundary_contrast=0.05,
        # boundary_levels=2,
        # boundary_ring=1.0,
        # boundary_samples=16,
        # boundary_tau=0.2,
        project=str(PROJECT),
        name=run_name,
    )


def find_best_weights() -> list[Path]:
    """Find best.pt files for this script's run names."""
    run_names = {f"{config.stem}_{MODEL_SCALE}" for config in CONFIGS_TO_RUN}
    best_paths = []
    for best_path in PROJECT.glob("*/weights/best.pt"):
        if best_path.parents[1].name in run_names:
            best_paths.append(best_path)
    return sorted(best_paths)


def run_test_inference() -> Path | None:
    """Evaluate trained checkpoints on the test split and save a summary CSV."""
    best_paths = find_best_weights()
    print(f"\nFound {len(best_paths)} trained checkpoints for testing")
    if not best_paths:
        return None

    results = []
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
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
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
        results.append(row)
        print(f"mAP50={row[1]:.4f} | mAP50-95={row[2]:.4f} | P={row[3]:.4f} | R={row[4]:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    TEST_PROJECT.mkdir(parents=True, exist_ok=True)
    summary_path = TEST_PROJECT / "test_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writerows(results)

    print(f"\nSaved test summary: {summary_path}")
    return TEST_PROJECT


def main() -> None:
    if RUN_PREPARE_DATA:
        prepare_data()

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

    if GENERATED_CONFIG_DIR.exists():
        shutil.rmtree(GENERATED_CONFIG_DIR)

    for config in CONFIGS_TO_RUN:
        train_config(config)

    run_test_inference()


if __name__ == "__main__":
    main()
