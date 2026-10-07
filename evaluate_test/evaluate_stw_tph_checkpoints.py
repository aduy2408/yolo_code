#!/usr/bin/env python3
"""Evaluation-only runner for STW-YOLO and TPH-YOLO checkpoints.

The runner never trains, downloads, uploads, or prepares datasets. Pass explicit
YOLO dataset YAMLs and checkpoint paths. Standard AP50 and mAP50-95 are read
from each backend's native evaluator. AP50-Small is computed by the shared
``evaluate_test.size_bucket_evaluator`` using a checkpoint loadable by the
project Ultralytics fork. For a legacy TPH checkpoint, provide
``--size-checkpoint`` with an equivalent converted checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("varroa", "tinyperson", "levirship")
IMAGE_SIZES = {"varroa": 640, "tinyperson": 640, "levirship": 512}


def parse_checkpoint(value: str) -> tuple[str, Path, Path | None]:
    model, sep, rest = value.partition("=")
    if not sep or not model:
        raise argparse.ArgumentTypeError("checkpoint must be MODEL=PATH")
    checkpoint, marker, size_checkpoint = rest.partition("::")
    if not checkpoint:
        raise argparse.ArgumentTypeError("checkpoint path is empty")
    return model, Path(checkpoint).expanduser().resolve(), Path(size_checkpoint).expanduser().resolve() if marker else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint,
                        help="MODEL=CHECKPOINT[::SIZE_CHECKPOINT], repeat for STW/TPH")
    parser.add_argument("--backend", choices=("stw", "tph"), required=True)
    parser.add_argument("--data-yaml", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--tph-root", type=Path, help="TPH-YOLO checkout, required with --backend tph")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def yaml_map(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        dataset, sep, path = value.partition("=")
        if dataset not in DATASETS or not sep or not path or dataset in result:
            raise ValueError(f"Invalid --data-yaml {value!r}")
        result[dataset] = Path(path).expanduser().resolve()
    missing = sorted(set(DATASETS) - result.keys())
    if missing:
        raise ValueError(f"Missing explicit dataset YAMLs: {missing}")
    for dataset, path in result.items():
        if not path.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist for {dataset}: {path}")
    return result


def project_ultralytics() -> None:
    project = str(ROOT / "models_related/ultralytics")
    sys.path.insert(0, project)


def evaluate_size(checkpoint: Path, data_yaml: Path, dataset: str, out_dir: Path, args: argparse.Namespace) -> dict[str, float | str]:
    project_ultralytics()
    from evaluate_test.size_bucket_evaluator import evaluate_native_size_buckets

    staged = out_dir / "weights" / "best.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists() or staged.is_symlink():
        staged.unlink()
    try:
        staged.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, staged)
    metrics: dict[str, float | str] = {}
    for split in ("val", "test"):
        metrics.update(evaluate_native_size_buckets(
            out_dir, data_yaml, split=split, imgsz=IMAGE_SIZES[dataset],
            batch=args.batch_size, device=args.device, workers=args.workers,
        ))
    return metrics


def evaluate_stw(checkpoint: Path, data_yaml: Path, dataset: str, out_dir: Path, args: argparse.Namespace) -> dict[str, float | str]:
    project_ultralytics()
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    metrics: dict[str, float | str] = {}
    for split in ("val", "test"):
        result = model.val(data=str(data_yaml), split=split, imgsz=IMAGE_SIZES[dataset],
                           batch=args.batch_size, device=args.device, workers=args.workers,
                           iou=0.5, plots=False, project=str(out_dir / "evaluation"),
                           name=f"standard_{split}", exist_ok=True)
        metrics[f"{split}/AP50"] = float(result.results_dict["metrics/mAP50(B)"])
        metrics[f"{split}/mAP50-95"] = float(result.results_dict["metrics/mAP50-95(B)"])
    return metrics


def evaluate_tph(checkpoint: Path, data_yaml: Path, dataset: str, out_dir: Path, args: argparse.Namespace) -> dict[str, float | str]:
    if args.tph_root is None:
        raise ValueError("--tph-root is required for --backend tph")
    metrics: dict[str, float | str] = {"nms_iou": 0.5}
    for split in ("val", "test"):
        command = [sys.executable, "val.py", "--data", str(data_yaml), "--weights", str(checkpoint),
                   "--img", str(IMAGE_SIZES[dataset]), "--batch-size", str(args.batch_size),
                   "--device", args.device, "--task", split, "--iou-thres", "0.5",
                   "--project", str(out_dir / "evaluation"), "--name", split, "--exist-ok", "--verbose"]
        completed = subprocess.run(command, cwd=args.tph_root, text=True, check=True, capture_output=True)
        lines = completed.stdout.splitlines() + completed.stderr.splitlines()
        summaries = [line.split() for line in lines if line.split() and line.split()[0] == "all" and len(line.split()) >= 7]
        if not summaries:
            raise RuntimeError(f"Unable to parse TPH val.py {split} summary")
        fields = summaries[-1]
        metrics[f"{split}/AP50"] = float(fields[-2])
        metrics[f"{split}/mAP50-95"] = float(fields[-1])
    return metrics


def main() -> None:
    args = parse_args()
    yamls = yaml_map(args.data_yaml)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluator = evaluate_stw if args.backend == "stw" else evaluate_tph
    rows = []
    for model_name, checkpoint, size_checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        for dataset in DATASETS:
            run_dir = args.output / model_name / dataset
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = evaluator(checkpoint, yamls[dataset], dataset, run_dir, args)
            size_source = size_checkpoint or checkpoint
            metrics.update(evaluate_size(size_source, yamls[dataset], dataset, run_dir, args))
            required = [f"{split}/{metric}" for split in ("val", "test") for metric in ("AP50", "mAP50-95")] + [f"{split}_size/AP50-Small" for split in ("val", "test")]
            missing = [key for key in required if key not in metrics]
            if missing:
                raise RuntimeError(f"Missing metrics for {model_name}/{dataset}: {missing}")
            record = {"model": model_name, "backend": args.backend, "dataset": dataset,
                      "checkpoint": str(checkpoint), "size_checkpoint": str(size_source), **metrics}
            (run_dir / "evaluation_metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    (args.output / "evaluation_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
