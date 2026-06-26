#!/usr/bin/env python3
"""Train the two TPH/KVCA Varroa configs: YOLOv8-style and YOLOv5-style."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "models_related" / "ultralytics"

CONFIGS = (
    ROOT / "models_related" / "models_config" / "yolov8" / "yolov8_varroa_tph_kvencoder_repc2f_ensimam_p2.yaml",
    ROOT / "models_related" / "models_config" / "yolov5" / "yolov5-tph-kvca-cbam-p2.yaml",
)


def prefer_local_ultralytics() -> None:
    """Force imports to use the repo's local Ultralytics fork."""
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


def prepare_data(root: Path, data_root: Path, out_dir: Path) -> Path:
    """Prepare YOLO dataset YAML when requested."""
    sys.path.insert(0, str(root))
    from prepare_dataset import prepare_dataset

    prepare_dataset(str(data_root), str(out_dir))
    data_yaml = out_dir / "varroa.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Prepared dataset YAML not found: {data_yaml}")
    return data_yaml


def config_for_scale(config: Path, scale: str, work_dir: Path) -> Path:
    """Copy a config and inject the requested Ultralytics scale."""
    text = config.read_text()
    lines = text.splitlines()
    out_lines = []
    inserted = False

    for line in lines:
        if line.startswith("scale:"):
            out_lines.append(f"scale: {scale}")
            inserted = True
        else:
            out_lines.append(line)
            if not inserted and line.startswith("nc:"):
                out_lines.append(f"scale: {scale}")
                inserted = True

    if not inserted:
        out_lines.insert(0, f"scale: {scale}")

    work_dir.mkdir(parents=True, exist_ok=True)
    scaled = work_dir / f"{config.stem}_{scale}{config.suffix}"
    scaled.write_text("\n".join(out_lines) + "\n")
    return scaled


def train_one(args: argparse.Namespace, config: Path, data_yaml: Path):
    """Train one config with matching pretrained weights."""
    from ultralytics import YOLO

    scaled_config = config_for_scale(config, args.scale, args.work_dir)
    run_name = f"{config.stem}_{args.scale}"

    print("\n" + "=" * 72)
    print(f"Training: {run_name}")
    print(f"Config:   {scaled_config}")
    print("=" * 72)

    model = YOLO(str(scaled_config))
    if args.pretrained:
        weights = args.pretrained
    elif "yolov5" in config.parts:
        weights = f"yolov5{args.scale}.pt"
    else:
        weights = f"yolov8{args.scale}.pt"

    if not args.no_pretrained:
        model.load(weights, smart_transfer=True)

    return model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        bbox_iou_loss=args.bbox_iou_loss,
        wiou_monotonous=args.wiou_monotonous,
        project=str(args.project),
        name=run_name,
        exist_ok=args.exist_ok,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLOv8 and YOLOv5 TPH/KVCA Varroa configs.")
    parser.add_argument("--scale", default="n", choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--configs", nargs="*", type=Path, default=list(CONFIGS), help="Override configs to train.")
    parser.add_argument("--data", type=Path, default=None, help="Existing dataset YAML. Required unless --prepare-data.")
    parser.add_argument("--prepare-data", action="store_true", help="Run prepare_dataset.py before training.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Raw data root for --prepare-data.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "datasets" / "varroa_yolo")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect" / "yolo_related" / "runs" / "train")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp") / "varroa_train_all_2_configs")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--bbox-iou-loss", default="wiou")
    parser.add_argument("--wiou-monotonous", action="store_true")
    parser.add_argument("--pretrained", default=None, help="Use one explicit pretrained checkpoint for all configs.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--clean-work-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefer_local_ultralytics()

    if args.clean_work_dir and args.work_dir.exists():
        shutil.rmtree(args.work_dir)

    if args.prepare_data:
        data_yaml = prepare_data(ROOT, args.data_root, args.out_dir)
    elif args.data:
        data_yaml = args.data
    else:
        data_yaml = args.out_dir / "varroa.yaml"

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}. Pass --data or use --prepare-data.")

    missing = [config for config in args.configs if not config.is_file()]
    if missing:
        raise FileNotFoundError("Missing config(s): " + ", ".join(str(p) for p in missing))

    for config in args.configs:
        train_one(args, config.resolve(), data_yaml.resolve())


if __name__ == "__main__":
    main()
