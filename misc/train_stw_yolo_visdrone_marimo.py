#!/usr/bin/env python3
"""Train STW-YOLO on the official VisDrone2019-DET split through Marimo."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STW_ROOT = Path(os.environ.get("STW_YOLO_ROOT", ROOT / "other_repo" / "STW-YOLO")).resolve()
SEED = 42
SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def convert(data_root: Path, out_root: Path) -> Path:
    images = out_root / "images"
    labels = out_root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for split, source_name in SPLITS.items():
        source = data_root / source_name
        if not source.is_dir():
            raise FileNotFoundError(f"Missing official VisDrone split: {source}")
        for image in sorted((source / "images").glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            target_image = images / split / image.name
            target_label = labels / split / f"{image.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                target_image.symlink_to(image)
            from PIL import Image
            with Image.open(image) as im:
                width, height = im.size
            rows = []
            ann = source / "annotations" / f"{image.stem}.txt"
            if ann.is_file():
                for line in ann.read_text(encoding="utf-8").splitlines():
                    fields = line.split(",")
                    if len(fields) < 8:
                        continue
                    x, y, w, h = map(int, fields[:4])
                    score, category = map(int, fields[4:6])
                    if score <= 0 or not 1 <= category <= 10 or w <= 0 or h <= 0:
                        continue
                    rows.append(f"{category - 1} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}")
            target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    yaml = out_root / "visdrone.yaml"
    yaml.write_text("\n".join([
        f"path: {out_root}", "train: images/train", "val: images/val", "test: images/test", "nc: 10", "names:",
        *[f"  {i}: {name}" for i, name in enumerate(NAMES)], "",
    ]), encoding="utf-8")
    return yaml


def upload(repo_id: str, run_dir: Path, remote: str, required: list[str]) -> None:
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{item}" for item in required}
    missing = sorted(expected - files)
    if missing:
        raise RuntimeError(f"Upload verification failed: {missing}")
    marker = {"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(run_dir / "upload_complete.json"), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/visdrone_stw")
    p.add_argument("--project", type=Path, default=ROOT / "runs/visdrone_stw")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf-repo-id", required=True)
    args = p.parse_args()
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    if not STW_ROOT.is_dir():
        raise FileNotFoundError(f"Missing STW-YOLO checkout: {STW_ROOT}")
    sys.path.insert(0, str(STW_ROOT))
    from ultralytics import YOLO
    data_yaml = convert(args.data_root.resolve(), args.dataset_root.resolve())
    run_dir = args.project.resolve() / "stw_yolo" / "mosaic" / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_yaml = STW_ROOT / "Lib/p2_rp5_yolo12s.yaml"
    model = YOLO("yolo12s.pt")
    model.train(data=str(data_yaml), model=str(model_yaml), epochs=args.epochs, imgsz=640, batch=args.batch_size, device=args.device, workers=args.workers, patience=args.patience, seed=SEED, deterministic=True, mosaic=1.0, close_mosaic=10, project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Missing checkpoint: {best}")
    metrics = {}
    for split in ("val", "test"):
        result = YOLO(str(best)).val(data=str(data_yaml), split=split, imgsz=640, batch=args.batch_size, device=args.device, workers=args.workers, iou=0.5, plots=False)
        values = result.results_dict
        metrics[f"{split}/AP50"] = float(values["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(values["metrics/mAP50-95(B)"])
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {"experiment_id": "visdrone-stw-yolo-mosaic-seed42", "baseline": {"config": str(model_yaml)}, "variant": {"augmentation": "default Ultralytics with mosaic=1.0, close_mosaic=10"}, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner": str(Path(__file__).resolve()), "python_executable": sys.executable, "dataset_root": str(args.data_root.resolve()), "dataset_yaml": str(data_yaml), "split_seed": 42, "training_seed": 42, "model/backbone/pretrained source": "STW-YOLO p2_rp5_yolo12s.yaml / yolo12s.pt", "image_size, batch_size, epochs, patience, AMP": [640, args.batch_size, args.epochs, args.patience, True], "NMS IoU": 0.5, "HF repo and remote prefix": [args.hf_repo_id, "runs/stw_yolo/mosaic/seed_42"], "required artifacts": ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "evaluation_metrics.json", "experiment_manifest.json"], "upload_required": True, **metrics}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    required = ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "evaluation_metrics.json", "experiment_manifest.json"]
    if any(not (run_dir / item).is_file() for item in required):
        raise RuntimeError("Required artifact missing")
    upload(args.hf_repo_id, run_dir, "runs/stw_yolo/mosaic/seed_42", required)


if __name__ == "__main__":
    main()
