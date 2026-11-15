#!/usr/bin/env python3
"""Prepare or run the unified PyTorch checkpoint evaluation queue.

The default action is setup-only: download URLs, queue commands, and a manifest
are written without launching evaluation. Use --run to execute sequentially.
The evaluator is intentionally configured for static TensorRT parity:
rect=False, batch=1, half=False, augment=False, NMS IoU=0.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CHECKPOINT_MANIFEST = ROOT / "manifests/checkpoints_manifest.json"
EVALUATOR = ROOT / "onnx_tensorrt_rerun_20261029/source/evaluate.py"

AUG_REPOS = {
    "oacp": "duyle2408/model-augmentation-seed42-oacp-runs",
    "copy_paste": "duyle2408/model-augmentation-seed42-copy-paste-runs",
}
AUG_MODELS = ("yolov8", "yolov11")
AUG_DATASETS = ("levir", "tinyperson", "varroa")
AUG_VARIANTS = (
    ("oacp", "r2"),
    ("copy_paste", "negative_canvas_r2"),
    ("copy_paste", "negative_canvas_r4"),
)
DATASET_NAMES = {"levir": "levirship", "tinyperson": "tinyperson", "varroa": "varroa"}
IMAGE_SIZES = {"levir": 512, "tinyperson": 640, "varroa": 640}
MOSAIC = {"levir": "no_mosaic", "tinyperson": "mosaic", "varroa": "mosaic"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_jobs(seeds: tuple[int, ...] = (42, 43)) -> list[dict[str, Any]]:
    manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    jobs = []
    for record in manifest["records"]:
        dataset = record["dataset"]
        for seed in seeds:
            jobs.append(
                {
                    "kind": "baseline",
                    "dataset": dataset,
                    "model": record["model"],
                    "seed": seed,
                    "repo_id": record["repo_id"],
                    "revision": record["revision"],
                    "hf_path": record["hf_path"].replace("/yolo11/", "/yolov11/").replace("seed_42", f"seed_{seed}"),
                    "expected_sha256": record["sha256"] if seed == 42 else None,
                    "image_size": 512 if dataset == "levirship" else 640,
                    "dataset_name": dataset,
                }
            )
    return jobs


def augmentation_jobs() -> list[dict[str, Any]]:
    jobs = []
    for model in AUG_MODELS:
        for dataset in AUG_DATASETS:
            for method, variant in AUG_VARIANTS:
                if dataset == "varroa" and method == "copy_paste":
                    continue
                repo_id = AUG_REPOS[method]
                jobs.append(
                    {
                        "kind": "augmentation",
                        "dataset": DATASET_NAMES[dataset],
                        "dataset_key": dataset,
                        "model": model,
                        "seed": 42,
                        "repo_id": repo_id,
                        "method": method,
                        "variant": variant,
                        "mosaic": MOSAIC[dataset],
                        "image_size": IMAGE_SIZES[dataset],
                        "hf_path": f"{model}/{dataset}/{method}/{variant}/{MOSAIC[dataset]}/seed_42/weights/best.pt",
                    }
                )
    return jobs


def all_jobs(seeds: tuple[int, ...] = (42, 43), include_augmentations: bool = True) -> list[dict[str, Any]]:
    jobs = baseline_jobs(seeds)
    if include_augmentations:
        jobs += augmentation_jobs()
    for index, job in enumerate(jobs):
        job["job_id"] = f"{index:03d}_{job['kind']}_{job['dataset']}_{job['model']}_s{job['seed']}"
    return jobs


def augmentation_jobs() -> list[dict[str, Any]]:
    jobs = []
    for model in AUG_MODELS:
        for dataset in AUG_DATASETS:
            for method, variant in AUG_VARIANTS:
                if dataset == "varroa" and method == "copy_paste":
                    continue
                repo_id = AUG_REPOS[method]
                jobs.append(
                    {
                        "kind": "augmentation",
                        "dataset": DATASET_NAMES[dataset],
                        "dataset_key": dataset,
                        "model": model,
                        "seed": 42,
                        "repo_id": repo_id,
                        "method": method,
                        "variant": variant,
                        "mosaic": MOSAIC[dataset],
                        "image_size": IMAGE_SIZES[dataset],
                        "hf_path": f"{model}/{dataset}/{method}/{variant}/{MOSAIC[dataset]}/seed_42/weights/best.pt",
                    }
                )
    return jobs




def checkpoint_path(output_root: Path, job: dict[str, Any]) -> Path:
    return output_root / "checkpoints" / job["job_id"] / "best.pt"


def result_path(output_root: Path, job: dict[str, Any]) -> Path:
    return output_root / "results" / job["job_id"]


def hf_url(job: dict[str, Any]) -> str:
    return f"https://huggingface.co/datasets/{job['repo_id']}/resolve/main/{job['hf_path']}"


def download(job: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    headers = {}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(hf_url(job), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def command(job: dict[str, Any], output_root: Path, dataset_yamls: dict[str, Path], python: str) -> list[str]:
    dataset = job["dataset"]
    return [
        python,
        str(EVALUATOR),
        "--model",
        str(checkpoint_path(output_root, job)),
        "--dataset",
        dataset,
        "--dataset-yaml",
        str(dataset_yamls[dataset]),
        "--split",
        "val",
        "test",
        "--results-root",
        str(result_path(output_root, job)),
        "--batch-size",
        "1",
        "--conf",
        "0.001",
        "--nms-iou",
        "0.5",
        "--max-det",
        "300",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset-yaml", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--baseline-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="download checkpoints and run evaluations")
    return parser.parse_args()


def parse_dataset_yamls(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in DATASET_NAMES.values() or not path:
            raise SystemExit(f"Invalid --dataset-yaml {value!r}; use dataset=path")
        result[key] = Path(path).expanduser().resolve()
    return result


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (ROOT / "results" / f"unified_pt_queue_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    output_root.mkdir(parents=True, exist_ok=False)
    jobs = all_jobs(tuple(args.baseline_seeds), include_augmentations=not args.baseline_only)
    dataset_yamls = parse_dataset_yamls(args.dataset_yaml)
    missing_yamls = sorted(set(DATASET_NAMES.values()) - dataset_yamls.keys())
    if args.run and missing_yamls:
        raise SystemExit(f"--run requires dataset YAMLs for: {', '.join(missing_yamls)}")

    for job in jobs:
        job["checkpoint"] = str(checkpoint_path(output_root, job))
        job["result_root"] = str(result_path(output_root, job))
        job["url"] = hf_url(job)
        job["command"] = command(job, output_root, dataset_yamls, args.python) if not missing_yamls else None

    (output_root / "queue_manifest.json").write_text(json.dumps({
        "protocol": {
            "backend": "unified_cloned_ultralytics",
            "rect": False,
            "batch_size": 1,
            "half": False,
            "augment": False,
            "conf": 0.001,
            "nms_iou": 0.5,
            "max_det": 300,
            "split_seed": 42,
        },
        "job_count": len(jobs),
        "jobs": jobs,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not missing_yamls:
        with (output_root / "commands.sh").open("w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            for job in jobs:
                handle.write(shlex.join(job["command"]) + "\n")

    if args.run:
        for number, job in enumerate(jobs, start=1):
            checkpoint = checkpoint_path(output_root, job)
            download(job, checkpoint)
            if job.get("expected_sha256") and sha256(checkpoint) != job["expected_sha256"]:
                raise SystemExit(f"SHA256 mismatch for {job['job_id']}: {checkpoint}")
            result = subprocess.run(job["command"], check=False)
            if result.returncode:
                raise SystemExit(f"job failed ({number}/{len(jobs)}): {job['job_id']}")
            print(f"DONE {number}/{len(jobs)} {job['job_id']}", flush=True)
    else:
        print(f"SETUP_ONLY output={output_root} jobs={len(jobs)}")
        print(f"MANIFEST {output_root / 'queue_manifest.json'}")
        if not missing_yamls:
            print(f"COMMANDS {output_root / 'commands.sh'}")


if __name__ == "__main__":
    main()
