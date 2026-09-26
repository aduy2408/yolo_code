#!/usr/bin/env python3
"""Run the matched VisDrone2019-DET YOLOv5-v11 baseline matrix.

The matrix contains the smallest standard detector scale for each family:

* YOLOv5n
* YOLOv8n
* YOLOv9t
* YOLOv10n
* YOLO11n

Each model is trained with seeds 42, 43, and 44 under both Mosaic and
no-Mosaic policies.  The detector YAMLs are pinned upstream configs.  The
runner is upload-required and must be launched through ``utils.marimo_ops``.
It intentionally requires an explicit VisDrone data root because the remote
mount is deployment-specific.
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

# Keep direct execution compatible with the repository-root imports.
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/ultralytics_upstream"

MODELS = {
    "yolov5n": (
        "yolov5nu.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v5/yolov5.yaml",
    ),
    "yolov8n": (
        "yolov8n.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v8/yolov8.yaml",
    ),
    "yolov9t": (
        "yolov9t.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v9/yolov9t.yaml",
    ),
    "yolov10n": (
        "yolov10n.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/v10/yolov10n.yaml",
    ),
    "yolo11n": (
        "yolo11n.pt",
        "vendor/ultralytics_upstream/ultralytics/cfg/models/11/yolo11.yaml",
    ),
}
SEEDS = (42, 43, 44)
AUGMENTATIONS = ("mosaic", "no_mosaic")
OPTIMIZER = "auto"
IMAGE_SIZE = 640
EXPECTED_SPLIT_IMAGES = {
    "VisDrone2019-DET-train": 6471,
    "VisDrone2019-DET-val": 548,
    "VisDrone2019-DET-test-dev": 1610,
}
VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
REQUIRED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
    "evaluation/test_size_ground_truth.json",
    "evaluation/test_size_predictions.json",
)
BASELINE_FORBIDDEN_MARKERS = (
    "CBAM",
    "ChannelAttention",
    "GAP",
    "FTAL",
    "OACP",
    "custom",
)


def local_ultralytics() -> None:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


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


def prepare_dataset(data_root: Path, output_dir: Path) -> Path:
    """Convert the official VisDrone folders to one reusable YOLO dataset."""
    from train_visdrone_scripts.train_all_visdrone_verifier import prepare_dataset as convert

    missing = [name for name in EXPECTED_SPLIT_IMAGES if not (data_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"VisDrone data root {data_root} is missing official split folders: {missing}"
        )
    for split, expected_count in EXPECTED_SPLIT_IMAGES.items():
        split_root = data_root / split
        image_count = sum(
            1
            for path in (split_root / "images").iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        annotation_count = sum(1 for path in (split_root / "annotations").glob("*.txt") if path.is_file())
        if image_count != expected_count or annotation_count != expected_count:
            raise RuntimeError(
                f"Noncanonical VisDrone split {split}: expected {expected_count} images and annotations, "
                f"got {image_count} images and {annotation_count} annotations"
            )
    return convert(data_root, output_dir).resolve()


def model_from_baseline_yaml(model_name: str):
    local_ultralytics()
    from ultralytics import YOLO

    weights_name, yaml_name = MODELS[model_name]
    yaml_path = (ROOT / yaml_name).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Pinned baseline YAML is missing: {yaml_path}")
    yaml_text = yaml_path.read_text(encoding="utf-8")
    forbidden = [marker for marker in BASELINE_FORBIDDEN_MARKERS if marker in yaml_text]
    if forbidden:
        raise RuntimeError(f"Refusing non-baseline YAML for {model_name}: {forbidden}")
    if "Detect" not in yaml_text:
        raise RuntimeError(f"Baseline YAML does not contain a standard Detect head: {yaml_path}")
    detect_lines = [line for line in yaml_text.splitlines() if "Detect" in line]
    if not any("P3" in line and "P4" in line and "P5" in line for line in detect_lines):
        raise RuntimeError(f"Refusing non-P3/P4/P5 detector YAML for {model_name}: {yaml_path}")

    # Generic v5/v8/v11 configs infer their compound scale from the filename.
    # The aliases point at the untouched pinned configs and do not change the
    # architecture.
    if model_name in {"yolov5n", "yolov8n", "yolo11n"}:
        alias_dir = ROOT / "runs/.baseline_yaml_aliases"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias = alias_dir / f"{model_name}.yaml"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(yaml_path)
        yaml_path = alias
    return YOLO(str(yaml_path)).load(weights_name), yaml_path


def patch_musgd_noncontiguous() -> None:
    """Work around pinned MuSGD's view assumption without editing the vendor."""
    local_ultralytics()
    from ultralytics.optim import muon

    if getattr(muon, "_project_noncontiguous_patch", False):
        return
    original = muon.muon_update

    def compatible(grad, momentum, beta=0.95, nesterov=True):
        single = hasattr(grad, "ndim")
        gradients = [grad] if single else list(grad)
        buffers = [momentum] if single else list(momentum)
        contiguous_gradients = [item.contiguous() for item in gradients]
        contiguous_buffers = [item.contiguous() for item in buffers]
        updates = original(
            contiguous_gradients,
            contiguous_buffers,
            beta=beta,
            nesterov=nesterov,
        )
        for target, source in zip(buffers, contiguous_buffers):
            target.copy_(source)
        return updates

    muon.muon_update = compatible
    muon._project_noncontiguous_patch = True


def metric_value(result: object, key: str) -> float:
    value = getattr(result, "results_dict", {}).get(key)
    if value is None:
        raise RuntimeError(f"Evaluator did not provide {key}")
    return float(value)


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    return (
        (run_dir / "weights/best.pt").is_file()
        and (run_dir / "weights/last.pt").is_file()
        and results.is_file()
        and sum(1 for _ in results.open(encoding="utf-8")) > 1
    )


def selected_jobs(models: list[str], seeds: list[int], augmentations: list[str], index: int, count: int):
    if count < 1 or not 0 <= index < count:
        raise ValueError("machine-index must be in [0, machine-count)")
    jobs = [(model, seed, augmentation) for augmentation in augmentations for model in models for seed in seeds]
    return [job for i, job in enumerate(jobs) if i % count == index]


def train_one(model_name: str, seed: int, augmentation: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / model_name / augmentation / f"seed_{seed}"
    if training_complete(run_dir):
        return run_dir
    if run_dir.exists():
        shutil.move(str(run_dir), str(run_dir.with_name(f"{run_dir.name}_incomplete_{int(time.time())}")))

    seed_everything(seed)
    model, _ = model_from_baseline_yaml(model_name)
    patch_musgd_noncontiguous()
    mosaic = 1.0 if augmentation == "mosaic" else 0.0
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
        optimizer=OPTIMIZER,
        mosaic=mosaic,
        close_mosaic=10 if mosaic else 0,
        plots=False,
        project=str(args.project / model_name / augmentation),
        name=f"seed_{seed}",
        exist_ok=True,
    )
    if not training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    local_ultralytics()
    from ultralytics import YOLO
    try:
        from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets
    except ModuleNotFoundError:
        from size_bucket_evaluator import evaluate_native_test_size_buckets

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, float | str] = {"nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            iou=0.5,
            plots=False,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics[f"{split}/AP50"] = metric_value(result, "metrics/mAP50(B)")
        metrics[f"{split}/mAP50-95"] = metric_value(result, "metrics/mAP50-95(B)")
    # Keep the same protocol used by the existing baseline matrix and the
    # post-hoc uploaded-metrics evaluator. These are native VisDrone test
    # images evaluated with the TinyBenchmark area buckets, not a substitute
    # for the standard test split metrics above.
    metrics.update(
        evaluate_native_test_size_buckets(
            run_dir,
            data_yaml,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
        )
    )
    metrics["test_size/dataset"] = "visdrone"
    return metrics


def upload_and_verify(api: object, repo_id: str, run_dir: Path, remote: str) -> None:
    missing = [path for path in REQUIRED_ARTIFACTS if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {run_dir}: {missing}")
    api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=repo_id, repo_type="dataset")
    files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    expected = {f"{remote}/{path}" for path in REQUIRED_ARTIFACTS}
    if not expected.issubset(files):
        raise RuntimeError(f"Upload verification failed for {remote}: {sorted(expected - files)}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": repo_id, "remote_prefix": remote, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
    api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/upload_complete.json", repo_id=repo_id, repo_type="dataset")


def verified_remote_prefixes(api: object, repo_id: str) -> set[str]:
    suffix = "/upload_complete.json"
    return {path[:-len(suffix)] for path in api.list_repo_files(repo_id, repo_type="dataset") if path.endswith(suffix)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--augmentations", nargs="+", choices=list(AUGMENTATIONS), default=list(AUGMENTATIONS))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/visdrone_baselines")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/visdrone_yolo_baselines")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--machine-count", type=int, default=1)
    parser.add_argument("--allow-subset", action="store_true", help="allow a selected subset of the baseline seeds")
    return parser.parse_args(argv)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.allow_subset and set(args.seeds) != set(SEEDS):
        raise ValueError("This baseline matrix requires exactly training seeds 42, 43, and 44")
    if os.environ.get("MARIMO_TRAIN_WORKFLOW") != "1":
        raise RuntimeError("Use python -m utils.marimo_ops launch for upload-required training")
    from utils.marimo_ops import ensure_hf_repo, require_training_context

    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    require_training_context(hf_repo_id=args.hf_repo_id)
    repo_id = ensure_hf_repo(args.hf_repo_id)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    data_yaml = prepare_dataset(args.data_root, args.dataset_root)
    jobs = selected_jobs(args.models, args.seeds, args.augmentations, args.machine_index, args.machine_count)
    verified = verified_remote_prefixes(api, repo_id)
    print(json.dumps({"jobs": len(jobs), "repo_id": repo_id, "data_yaml": str(data_yaml)}, sort_keys=True), flush=True)

    for model_name, seed, augmentation in jobs:
        remote = f"runs/{model_name}/{augmentation}/seed_{seed}"
        if remote in verified:
            print(f"SKIP_VERIFIED {remote}", flush=True)
            continue
        run_dir = train_one(model_name, seed, augmentation, data_yaml, args)
        metrics = evaluate(run_dir, data_yaml, args)
        manifest = {
            "dataset": "VisDrone2019-DET",
            "data_root": str(args.data_root),
            "data_yaml": str(data_yaml),
            "official_split": "VisDrone2019-DET-train / val / test-dev",
            "model": model_name,
            "pretrained": MODELS[model_name][0],
            "model_yaml": str((ROOT / MODELS[model_name][1]).resolve()),
            "seed": seed,
            "split_seed": None,
            "split_provenance": "official VisDrone2019-DET split, no random reassignment",
            "augmentation": augmentation,
            "mosaic": 1.0 if augmentation == "mosaic" else 0.0,
            "close_mosaic": 10 if augmentation == "mosaic" else 0,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "optimizer": OPTIMIZER,
            "optimizer_expected": "MuSGD when Ultralytics auto exceeds its iteration threshold",
            "batch_size": args.batch_size,
            "workers": args.workers,
            "nms_iou": 0.5,
            "git_sha": git_sha(),
            "hf_repo_id": repo_id,
            "machine_index": args.machine_index,
            "machine_count": args.machine_count,
            "test_protocol": "Ultralytics native VisDrone2019-DET-test-dev split",
            "test_size_protocol": metrics["test_size/protocol"],
            "test_size_source_artifacts": [
                "evaluation/test_size_ground_truth.json",
                "evaluation/test_size_predictions.json",
            ],
            **metrics,
        }
        (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        upload_and_verify(api, repo_id, run_dir, remote)
        print(f"COMPLETE {remote}", flush=True)


if __name__ == "__main__":
    main()
