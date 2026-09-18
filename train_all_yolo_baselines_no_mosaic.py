#!/usr/bin/env python3
"""Train the five standard YOLO baselines on three datasets without Mosaic.

This is an upload-required Marimo runner. It keeps the detector and all
non-Mosaic training settings matched across datasets, fixes the data split
seed at 42, and varies only the requested training seeds.
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
UPSTREAM_ULTRALYTICS = ROOT / "vendor/ultralytics_upstream"
MODELS = {
    "yolov5": ("yolov5nu.pt", "vendor/ultralytics_upstream/ultralytics/cfg/models/v5/yolov5.yaml"),
    "yolov8": ("yolov8n.pt", "vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml"),
    "yolov9": ("yolov9t.pt", "vendor/ultralytics_upstream/ultralytics/cfg/models/v9/yolov9t.yaml"),
    "yolov10": ("yolov10n.pt", "vendor/ultralytics_upstream/ultralytics/cfg/models/v10/yolov10n.yaml"),
    "yolov11": ("yolo11n.pt", "vendor/ultralytics_upstream/ultralytics/cfg/models/11/yolo11.yaml"),
}
SEEDS = (42, 43, 44)
SPLIT_SEED = 42
DATASETS = ("varroa", "tinyperson", "levirship")
DEFAULT_DATA_ROOTS = {
    "varroa": "/marimo/Varroa",
    "tinyperson": "/marimo/TinyPersonData",
    "levirship": "/marimo/LevirShip/LevirShipData",
}
IMAGE_SIZES = {"varroa": 640, "tinyperson": 640, "levirship": 512}
REQUIRED = (
    "weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
    "evaluation_metrics.json", "experiment_manifest.json",
)


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


def local_ultralytics() -> None:
    """Use the pinned upstream package shipped as the repository submodule."""
    path = str(UPSTREAM_ULTRALYTICS)
    if path not in sys.path:
        sys.path.insert(0, path)


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt")) and results.is_file() and sum(1 for _ in results.open(encoding="utf-8")) > 1


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare_dataset(name: str, data_root: Path, dataset_root: Path) -> Path:
    if name == "varroa":
        from misc.prepare_dataset import prepare_dataset as prepare_varroa
        return prepare_varroa(
            data_root, dataset_root / "varroa_split_42", gt_source="gt_one",
            only_positives=True, class_policy="map-3-to-1", seed=SPLIT_SEED,
        ).resolve()
    if name == "levirship":
        from misc.prepare_levir_ship import prepare
        return prepare(data_root, dataset_root / "levirship_split_42", SPLIT_SEED).resolve()
    import train_all_tinyperson as tiny
    test_dir = tiny.prepare_test_set(data_root, dataset_root)
    return tiny.prepare_seed_dataset(data_root, dataset_root, test_dir, SPLIT_SEED).joinpath("tinyperson.yaml").resolve()


def model_from_baseline_yaml(model_name: str):
    """Construct from the pinned upstream baseline YAML, then load n/t weights."""
    local_ultralytics()
    from ultralytics import YOLO

    weights_name, yaml_name = MODELS[model_name]
    yaml_path = ROOT / yaml_name
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Pinned baseline YAML is missing: {yaml_path}")
    # Ultralytics infers the compound scale from the filename. Keep the
    # canonical YAML content untouched and expose it through an n-named
    # symlink for the generic v5/v8/v11 files. The symlink is the effective
    # model YAML, not a custom architecture.
    if model_name in {"yolov5", "yolov8", "yolov11"}:
        scaled = ROOT / "runs" / ".baseline_yaml_aliases" / f"{model_name}n.yaml"
        scaled.parent.mkdir(parents=True, exist_ok=True)
        if scaled.exists() or scaled.is_symlink():
            scaled.unlink()
        scaled.symlink_to(yaml_path)
        yaml_path = scaled
    return YOLO(str(yaml_path)).load(weights_name), yaml_path


def metric_value(result: object, key: str) -> float:
    value = getattr(result, "results_dict", {}).get(key)
    if value is None:
        raise RuntimeError(f"Evaluator did not provide {key}")
    return float(value)


def selected_jobs(datasets: list[str], models: list[str], seeds: list[int], machine_index: int, machine_count: int):
    """Return a disjoint deterministic shard of the full dataset/model/seed matrix."""
    if machine_count < 1 or not 0 <= machine_index < machine_count:
        raise ValueError("machine-index must be in [0, machine-count)")
    jobs = [(dataset, model, seed) for dataset in datasets for model in models for seed in seeds]
    return [job for index, job in enumerate(jobs) if index % machine_count == machine_index]


def evaluate_standard(run_dir: Path, data_yaml: Path, dataset: str, args: argparse.Namespace) -> dict[str, float]:
    local_ultralytics()
    from ultralytics import YOLO
    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=IMAGE_SIZES[dataset],
            batch=args.batch_size, device=args.device, workers=args.workers,
            iou=0.5, plots=False, project=str(run_dir / "evaluation"),
            name=split, exist_ok=True,
        )
        metrics[f"{split}/AP50"] = metric_value(result, "metrics/mAP50(B)")
        metrics[f"{split}/mAP50-95"] = metric_value(result, "metrics/mAP50-95(B)")
    return metrics


def evaluate_tinyperson(run_dir: Path, data_yaml: Path, data_root: Path, args: argparse.Namespace) -> dict[str, float]:
    import train_all_tinyperson as tiny
    # The established evaluator emits split-qualified val and merged-test
    # metrics and records the official corner-window protocol.
    tiny_args = argparse.Namespace(**vars(args))
    tiny_args.imgsz = IMAGE_SIZES["tinyperson"]
    test_dir = tiny.prepare_test_set(data_root, args.dataset_root)
    tiny.evaluate(run_dir, data_yaml, test_dir, data_root, tiny_args)
    raw = json.loads((run_dir / "evaluation_metrics.json").read_text())
    # The shared evaluator uses Ultralytics' raw names. Normalize them here so
    # every dataset follows the project-wide split-qualified reporting contract.
    metrics = dict(raw)
    for split in ("val", "test"):
        metrics[f"{split}/AP50"] = float(raw[f"{split}/metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(raw[f"{split}/metrics/mAP50-95(B)"])
    required = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"TinyPerson evaluator missing split-qualified metrics: {missing}")
    return {key: float(metrics[key]) for key in required} | metrics


def train_one(dataset: str, model_name: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / dataset / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not training_complete(run_dir):
        if run_dir.exists():
            archive = run_dir.with_name(f"{run_dir.name}_incomplete_{int(time.time())}")
            shutil.move(str(run_dir), str(archive))
        seed_everything(seed)
        model, _ = model_from_baseline_yaml(model_name)
        kwargs = dict(
            data=str(data_yaml), epochs=args.epochs, imgsz=IMAGE_SIZES[dataset],
            batch=args.batch_size, device=args.device, workers=args.workers,
            patience=args.patience, seed=seed, deterministic=True, amp=True,
            optimizer="SGD",
            mosaic=0.0, close_mosaic=0, plots=False,
            project=str(args.project / dataset / model_name),
            name=f"seed_{seed}", exist_ok=True,
        )
        model.train(**kwargs)
    if not training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def upload_and_verify(api: object, repo_id: str, run_dir: Path, remote: str) -> None:
    missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {run_dir}: {missing}")
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{path}" for path in REQUIRED}
    if not expected.issubset(files):
        raise RuntimeError(f"Upload verification failed for {remote}: {sorted(expected - files)}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", action="append", metavar="DATASET=PATH", help="Override a dataset root")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/no_mosaic_baselines")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/yolo_baselines_no_mosaic")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--machine-index", type=int, default=0, help="Zero-based shard index")
    parser.add_argument("--machine-count", type=int, default=1, help="Total number of cooperating servers")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required training")
    from utils.marimo_ops import ensure_hf_repo, require_training_context
    args.dataset_root, args.project = args.dataset_root.resolve(), args.project.resolve()
    if args.machine_count < 1 or not 0 <= args.machine_index < args.machine_count:
        raise ValueError("machine-index must be in [0, machine-count)")
    roots = dict(DEFAULT_DATA_ROOTS)
    for item in args.data_root or []:
        dataset, sep, path = item.partition("=")
        if dataset not in DATASETS or not sep or not path:
            raise ValueError(f"Invalid --data-root {item!r}; use DATASET=PATH")
        roots[dataset] = path
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    token = os.environ["HF_TOKEN"]
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    jobs = selected_jobs(args.datasets, args.models, args.seeds, args.machine_index, args.machine_count)
    print(json.dumps({"machine_index": args.machine_index, "machine_count": args.machine_count, "jobs": len(jobs)}, sort_keys=True), flush=True)
    for dataset, model_name, seed in jobs:
        data_yaml = prepare_dataset(dataset, Path(roots[dataset]), args.dataset_root)
        run_dir = train_one(dataset, model_name, seed, data_yaml, args)
        metrics = evaluate_tinyperson(run_dir, data_yaml, Path(roots[dataset]), args) if dataset == "tinyperson" else evaluate_standard(run_dir, data_yaml, dataset, args)
        manifest = {
            "dataset": dataset, "model": model_name, "pretrained": MODELS[model_name][0],
            "model_yaml": str((ROOT / MODELS[model_name][1]).resolve()),
            "seed": seed, "split_seed": SPLIT_SEED, "mosaic": 0.0, "close_mosaic": 0,
            "epochs": args.epochs, "patience": args.patience, "imgsz": IMAGE_SIZES[dataset],
            "batch_size": args.batch_size, "workers": args.workers, "nms_iou": 0.5,
            "data_yaml": str(data_yaml), "git_sha": git_sha(), "hf_repo_id": repo_id,
            "machine_index": args.machine_index, "machine_count": args.machine_count,
            "test_protocol": "TinyPerson official corner-window merged test" if dataset == "tinyperson" else "Ultralytics test split",
            **metrics,
        }
        (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        upload_and_verify(api, repo_id, run_dir, f"runs/{dataset}/{model_name}/seed_{seed}")
        print(f"COMPLETE {dataset}/{model_name}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
