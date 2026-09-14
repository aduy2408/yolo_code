#!/usr/bin/env python3
"""Run the STW-YOLO model sequentially on the three Marimo-mounted datasets.

This runner is intentionally upload-required and restart-safe at the dataset level.
It prepares each dataset once, trains seed 42 for 100 epochs, evaluates at NMS IoU
0.5, uploads the required artifacts, verifies the remote paths, then advances.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STW_ROOT = Path(os.environ.get("STW_YOLO_ROOT", ROOT / "STW-YOLO")).resolve()
if str(STW_ROOT) not in sys.path:
    sys.path.insert(0, str(STW_ROOT))
SEED = 42
SPLIT_SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
WORKERS = 8
NMS_IOU = 0.5
DEFAULT_HF_REPO = "duyle2408/stw-yolo-runs"

DATA_ROOTS = {
    "varroa": Path("/marimo/Varroa"),
    "levirship": Path("/marimo/LevirShip/LevirShipData"),
    "tinyperson": Path("/marimo/TinyPerson"),
}
IMAGE_SIZES = {"varroa": 640, "levirship": 512, "tinyperson": 640}


def git_sha(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def symlink_dataset(source: Path, destination: Path, split: str) -> None:
    image_root = source / split / "videos"
    label_root = source / split / "labels"
    out_images = destination / "images" / split
    out_labels = destination / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for scene in sorted(image_root.iterdir()):
        if not scene.is_dir():
            continue
        for image in scene.glob("*.png"):
            label = label_root / scene.name / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing Varroa label for {image}")
            (out_images / image.name).symlink_to(image)
            (out_labels / label.name).symlink_to(label)


def prepare_varroa(source: Path, runtime: Path) -> Path:
    from misc.prepare_dataset import prepare_dataset

    return prepare_dataset(
        source,
        runtime,
        gt_source="gt_one",
        only_positives=True,
        class_policy="map-3-to-1",
        seed=SPLIT_SEED,
    )


def prepare_levir(source: Path, runtime: Path) -> Path:
    from misc.prepare_levir_ship import prepare

    return prepare(source, runtime, seed=SPLIT_SEED)


def prepare_tinyperson(source: Path, runtime: Path) -> Path:
    import train_all_tinyperson as tiny

    test_dir = tiny.prepare_test_set(source, runtime)
    del test_dir
    seed_dir = tiny.prepare_seed_dataset(source, runtime, test_dir, SPLIT_SEED)
    return seed_dir / "tinyperson.yaml"


def prepare_dataset(name: str, runtime_root: Path) -> Path:
    source = DATA_ROOTS[name]
    if not source.is_dir():
        raise FileNotFoundError(f"Remote dataset root is missing: {source}")
    runtime = runtime_root / name
    if name == "varroa":
        return prepare_varroa(source, runtime)
    if name == "levirship":
        return prepare_levir(source, runtime)
    return prepare_tinyperson(source, runtime)


def upload_and_verify(repo_id: str, run_dir: Path, remote_prefix: str, required: list[str]) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        path_in_repo=remote_prefix,
        repo_id=repo_id,
        repo_type="dataset",
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote_prefix}/{path}" for path in required}
    missing = sorted(expected - remote_files)
    if missing:
        raise RuntimeError(f"Remote verification failed for {remote_prefix}: {missing}")
    marker = {
        "repo_id": repo_id,
        "remote_prefix": remote_prefix,
        "verified_paths": sorted(expected),
    }
    (run_dir / "upload_complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(run_dir / "upload_complete.json"),
        path_in_repo=f"{remote_prefix}/upload_complete.json",
        repo_id=repo_id,
        repo_type="dataset",
    )


def run_one(name: str, args: argparse.Namespace, data_yaml: Path, repo_id: str) -> None:
    from ultralytics import YOLO

    image_size = IMAGE_SIZES[name]
    run_dir = args.project / name / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "evaluation_metrics.json",
        "experiment_manifest.json",
    ]
    if (run_dir / "upload_complete.json").is_file():
        print(f"Skipping verified run: {run_dir}", flush=True)
        return

    best = run_dir / "weights/best.pt"
    results = run_dir / "results.csv"
    if not best.is_file() or not results.is_file():
        model_yaml = STW_ROOT / "Lib/p2_rp5_yolo12s.yaml"
        model = YOLO("yolo12s.pt")
        model.train(
            data=str(data_yaml),
            model=str(model_yaml),
            epochs=EPOCHS,
            imgsz=image_size,
            batch=BATCH_SIZE,
            device=args.device,
            workers=WORKERS,
            patience=0,
            seed=SEED,
            deterministic=True,
            project=str(args.project / name),
            name=f"seed_{SEED}",
            exist_ok=True,
        )
    if not best.is_file():
        raise RuntimeError(f"Missing trained checkpoint: {best}")
    model_yaml = STW_ROOT / "Lib/p2_rp5_yolo12s.yaml"
    validation = YOLO(str(best)).val(
        data=str(data_yaml),
        split="test",
        imgsz=image_size,
        batch=BATCH_SIZE,
        device=args.device,
        workers=WORKERS,
        iou=NMS_IOU,
        plots=False,
    )
    metrics = getattr(validation, "results_dict", {})
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n", encoding="utf-8")
    manifest = {
        "dataset": name,
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "data_yaml": str(data_yaml),
        "model_yaml": str(model_yaml),
        "pretrained": "yolo12s.pt",
        "epochs": EPOCHS,
        "patience": 0,
        "imgsz": image_size,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "nms_iou": NMS_IOU,
        "source_commit": git_sha(ROOT),
        "stw_yolo_commit": git_sha(STW_ROOT),
        "hf_repo_id": repo_id,
        "remote_prefix": f"runs/{name}/seed_{SEED}",
        "upload_required": True,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    missing = [path for path in required if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Missing required artifacts for {name}: {missing}")
    upload_and_verify(repo_id, run_dir, f"runs/{name}/seed_{SEED}", required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATA_ROOTS), default=sorted(DATA_ROOTS))
    parser.add_argument("--project", type=Path, default=ROOT / "runs/stw_yolo_marimo")
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "scratch/stw_yolo_marimo")
    parser.add_argument("--device", default="0")
    parser.add_argument("--hf-repo-id", default=os.environ.get("MARIMO_HF_REPO_ID", DEFAULT_HF_REPO))
    return parser.parse_args()


def main() -> None:
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use utils.marimo_ops launch for upload-required Marimo training")
    args = parse_args()
    args.project = args.project.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        data_yaml = prepare_dataset(name, args.runtime_root)
        run_one(name, args, data_yaml, args.hf_repo_id)


if __name__ == "__main__":
    main()
