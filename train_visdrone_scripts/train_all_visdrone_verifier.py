#!/usr/bin/env python3
"""Train VisDrone2019 dataset using the CandidateVerifier models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG = ROOT / "models_related/models_config/yolov8/visdrone/yolov8n_p2p3p4_visdrone_plain_gap.yaml"


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


def convert_visdrone_labels(source_dir: Path, images_out_dir: Path, labels_out_dir: Path, split: str) -> None:
    """Convert VisDrone annotations to YOLO format with class normalization."""
    from PIL import Image

    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # Convert annotations
    annotations_dir = source_dir / "annotations"
    images_in_dir = source_dir / "images"

    if not annotations_dir.exists() or not images_in_dir.exists():
        raise FileNotFoundError(f"Missing annotations or images under {source_dir}")

    annotation_files = list(annotations_dir.glob("*.txt"))
    print(f"Converting {len(annotation_files)} labels for split '{split}'...")

    for f in annotation_files:
        img_name = f.with_suffix(".jpg").name
        src_img_path = images_in_dir / img_name
        dest_img_path = images_out_dir / img_name

        if not src_img_path.exists():
            # Try png fallback if jpg not present
            img_name = f.with_suffix(".png").name
            src_img_path = images_in_dir / img_name
            dest_img_path = images_out_dir / img_name
            if not src_img_path.exists():
                continue

        # Symlink or copy image
        if not dest_img_path.exists():
            try:
                dest_img_path.symlink_to(src_img_path)
            except OSError:
                shutil.copy2(src_img_path, dest_img_path)

        # Process annotations
        img_size = Image.open(src_img_path).size
        dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]
        lines = []

        with open(f, "r", encoding="utf-8") as file:
            for line in file:
                row = [x.strip() for x in line.split(",") if x.strip()]
                if len(row) < 6:
                    continue
                # Skip ignored regions: row[4] is score/ignored. 0 means ignored.
                if row[4] != "0":
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1  # classes 1-10 -> 0-9
                    # Map any class out of 0-9 boundary (if 11 - 'others') to ignore
                    if cls < 0 or cls > 9:
                        continue
                    # Convert to YOLO format [x_center, y_center, width, height] normalized
                    x_center = (x + w / 2.0) * dw
                    y_center = (y + h / 2.0) * dh
                    w_norm = w * dw
                    h_norm = h * dh
                    lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        with open(labels_out_dir / f.name, "w", encoding="utf-8") as out_file:
            out_file.writelines(lines)


def prepare_dataset(data_root: Path, output_dir: Path) -> Path:
    """Prepares the VisDrone dataset and returns the path to the dataset YAML file."""
    output_dir = output_dir.resolve()
    print(f"Preparing VisDrone dataset under: {output_dir}")

    splits = {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }

    for source_folder, split_name in splits.items():
        src_path = data_root / source_folder
        if not src_path.exists():
            raise FileNotFoundError(f"Source VisDrone folder '{src_path}' not found. Please ensure dataset is extracted.")

        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        # Perform conversion
        convert_visdrone_labels(src_path, img_out, lbl_out, split_name)

    # Create dataset yaml file
    yaml_path = output_dir / "visdrone.yaml"
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Dataset YAML generated at: {yaml_path}")
    return yaml_path


def train(args: argparse.Namespace, data_yaml: Path, seed: int) -> None:
    local_ultralytics()
    from ultralytics import YOLO

    seed_everything(seed)
    run_name = f"visdrone_{args.verifier_mode}_seed{seed}"
    run_dir = args.project / run_name

    model = YOLO(str(CONFIG))

    # Mandatory YOLO evaluation protocol: NMS IoU explicitly set to 0.5
    model.train(
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
        project=str(args.project),
        name=run_name,
        exist_ok=True,
        # FTAL (Factorized TAL) settings (fixed across experiments)
        factorized_tal_target=True,
        factorized_tal_tau=0.75,
        factorized_tal_kappa=1.5,
        factorized_tal_lambda=0.5,
        factorized_tal_s_max=32.0,
        factorized_tal_warmup_start=5,
        factorized_tal_warmup_end=15,
        factorized_tal_p2_only=True,
        # Candidate Verifier settings
        verifier_mode=args.verifier_mode,
        verifier_alpha=args.verifier_alpha,
        verifier_loss_gain=args.verifier_loss_gain,
    )

    # Perform validation and test evaluations explicitly setting NMS IoU threshold to 0.5
    print("Evaluating validation set (NMS IoU = 0.5)...")
    val_results = model.val(iou=0.5, split="val")

    print("Evaluating test set (NMS IoU = 0.5)...")
    test_results = model.val(iou=0.5, split="test")

    # Save metrics JSON
    metrics = {
        "nms_iou": 0.5,
        "verifier_mode": args.verifier_mode,
        "verifier_alpha": args.verifier_alpha,
        "verifier_loss_gain": args.verifier_loss_gain,
        "val": {key: float(value) for key, value in val_results.results_dict.items()},
        "test": {key: float(value) for key, value in test_results.results_dict.items()},
    }
    metrics_path = run_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics saved to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/varroa/VisDrone2019"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/data/varroa/VisDrone2019/yolo_format"))
    parser.add_argument("--project", type=Path, default=ROOT / "runs/visdrone_verifiers")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verifier-mode", choices=["a1_box_fovea", "a3_semantic_structural", "a4_raw_adapted", "none"], default="none")
    parser.add_argument("--verifier-alpha", type=float, default=0.5)
    parser.add_argument("--verifier-loss-gain", type=float, default=0.5)
    args = parser.parse_args()

    # Prepare dataset
    data_yaml = prepare_dataset(args.data_root, args.output_dir)

    # Run training
    train(args, data_yaml, args.seed)


if __name__ == "__main__":
    main()
