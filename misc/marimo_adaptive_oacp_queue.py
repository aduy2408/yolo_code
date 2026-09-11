#!/usr/bin/env python3
"""Sequential six-run queue for mass/load/spacing adaptive OACP ablations.

Runs one LEVIR-Ship and one TinyPerson job for each adaptive policy. Every job
is launched through utils.marimo_ops, evaluated, uploaded, and verified before
the next job starts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from utils.marimo_ops import launch_detached, status

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python")
VARIANTS = ("mass_adaptive", "load_adaptive", "spacing_adaptive")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
)


def _job(root: Path, dataset: str, variant: str, repo: str) -> dict[str, object]:
    if dataset == "levir":
        project = root / "runs" / f"levir_adaptive_oacp_{variant}_seed42"
        run_dir = project / "p2p3p4_oacp"
        command = [
            PYTHON, "misc/train_context_aug_matrix.py",
            "--data-root", "/marimo/LevirShip/LevirShipData",
            "--dataset-root", str(root / "datasets/levir_ship_seed42"),
            "--project", str(project), "--hf-repo-id", repo,
            "--only", "p2p3p4_oacp", "--seed", "42",
            "--epochs", "100", "--patience", "0", "--imgsz", "512",
            "--batch-size", "8", "--workers", "4", "--device", "cuda",
            "--mosaic", "1.0", "--close-mosaic", "10",
        ]
    else:
        project = root / "runs" / f"tinyperson_adaptive_oacp_{variant}_seed42"
        run_dir = project / f"{variant}_oacp_mosaic" / "seed_42_corner_sw640_sh512"
        command = [
            PYTHON, "train_all_tinyperson_yolo11_oacp_mosaic_matrix.py",
            "--data-root", "/marimo/TinyPerson",
            "--dataset-root", str(root / "datasets/tinyperson_seed42"),
            "--project", str(project), "--hf-repo-id", repo,
            "--variants", f"{variant}_oacp_mosaic",
            "--model", "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3p4_plain.yaml",
            "--seeds", "42", "--split-seed", "42", "--epochs", "100",
            "--patience", "0", "--imgsz", "640", "--batch-size", "8",
            "--workers", "4", "--device", "cuda", "--amp", "--confirm-settings",
        ]
    return {
        "name": f"{dataset}_{variant}_seed42",
        "dataset": dataset,
        "variant": variant,
        "repo": repo,
        "job_dir": root / "jobs" / f"adaptive_{dataset}_{variant}_seed42",
        "run_dir": run_dir,
        "command": command,
        "env": {
            "OACP_VARIANT": variant,
            "YOLO_CONTEXT_AUG": "oacp",
            "YOLO_LEGACY_DOUBLE_OACP": "0",
        },
    }


def _complete(job: dict[str, object]) -> bool:
    run_dir = Path(job["run_dir"])
    return all((run_dir / name).is_file() for name in REQUIRED) and (run_dir / "upload_complete.json").is_file()


def _wait(job: dict[str, object], poll_seconds: int) -> None:
    job_dir = Path(job["job_dir"])
    while True:
        info = status(job_dir)
        if not info["process_alive"]:
            if not _complete(job):
                raise RuntimeError(f"Job stopped without verified completion: {job['name']}")
            return
        time.sleep(poll_seconds)


def run(args: argparse.Namespace) -> None:
    repos = {
        "levir_mass_adaptive": args.levir_mass_adaptive,
        "levir_load_adaptive": args.levir_load_adaptive,
        "levir_spacing_adaptive": args.levir_spacing_adaptive,
        "tiny_mass_adaptive": args.tiny_mass_adaptive,
        "tiny_load_adaptive": args.tiny_load_adaptive,
        "tiny_spacing_adaptive": args.tiny_spacing_adaptive,
    }
    jobs = []
    for variant in VARIANTS:
        jobs.append(_job(args.root, "levir", variant, repos[f"levir_{variant}"]))
        jobs.append(_job(args.root, "tinyperson", variant, repos[f"tiny_{variant}"]))
    for job in jobs:
        job_dir = Path(job["job_dir"])
        if _complete(job):
            print(f"SKIP_VERIFIED {job['name']}", flush=True)
            continue
        if status(job_dir)["process_alive"]:
            raise RuntimeError(f"Already running: {job['name']}")
        env = {**job["env"], "HF_TOKEN": os.environ["HF_TOKEN"], "MARIMO_HF_REPO_ID": job["repo"]}
        launch_detached(
            job["command"], cwd=args.root, log_path=job_dir / "train.log",
            pid_path=job_dir / "train.pid", state_path=job_dir / "state.json", env=env,
        )
        _wait(job, args.poll_seconds)
        print(f"COMPLETE {job['name']} repo={job['repo']}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    for dataset in ("levir", "tiny"):
        for variant in VARIANTS:
            parser.add_argument(f"--{dataset}-{variant.replace('_', '-')}", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
