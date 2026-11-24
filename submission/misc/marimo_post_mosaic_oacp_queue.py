#!/usr/bin/env python3
"""Sequential matched LEVIR post-Mosaic OACP placement/signal runs."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from utils.marimo_ops import launch_detached, status

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "manifest.json",
)


def jobs(root: Path, repo: str) -> list[dict[str, object]]:
    def command(project: Path) -> list[str]:
        return [
            PYTHON,
            "misc/train_context_aug_matrix.py",
            "--data-root", "/marimo/LevirShip/LevirShipData",
            "--dataset-root", str(root / "datasets/levir_ship_seed42"),
            "--project", str(project),
            "--hf-repo-id", repo,
            "--only", "p2p3p4_oacp",
            "--seed", "42", "--split-seed", "42",
            "--epochs", "100", "--patience", "0",
            "--imgsz", "512", "--batch-size", "8", "--workers", "8",
            "--device", "cuda", "--mosaic", "1.0", "--close-mosaic", "10",
        ]

    run1 = root / "runs/levir_post_mosaic_oacp_run1_load"
    run2 = root / "runs/levir_post_mosaic_oacp_run2_spatial"
    return [
        {
            "name": "post_mosaic_load_adaptive_seed42",
            "run_dir": run1 / "p2p3p4_oacp",
            "job_dir": root / "jobs/levir_post_mosaic_load_adaptive_seed42",
            "command": command(run1),
            "env": {
                "OACP_VARIANT": "load_adaptive",
                "YOLO_CONTEXT_AUG": "oacp",
                "YOLO_LEGACY_DOUBLE_OACP": "0",
                "OACP_PLACEMENT": "post_mosaic",
                "OACP_REMOTE_SUFFIX": "post_mosaic_load_adaptive",
            },
        },
        {
            "name": "post_mosaic_spatial_load_adaptive_seed42",
            "run_dir": run2 / "p2p3p4_oacp",
            "job_dir": root / "jobs/levir_post_mosaic_spatial_load_adaptive_seed42",
            "command": command(run2),
            "env": {
                "OACP_VARIANT": "spatial_load_adaptive",
                "YOLO_CONTEXT_AUG": "oacp",
                "YOLO_LEGACY_DOUBLE_OACP": "0",
                "OACP_SPATIAL_LOAD_MIN": "0.0",
                "OACP_SPATIAL_LOAD_MAX": "0.04",
                "OACP_PLACEMENT": "post_mosaic",
                "OACP_REMOTE_SUFFIX": "post_mosaic_spatial_load_adaptive",
            },
        },
    ]


def complete(run_dir: Path) -> bool:
    return all((run_dir / item).is_file() for item in REQUIRED) and (run_dir / "upload_complete.json").is_file()


def wait_for(job: dict[str, object], poll_seconds: int) -> None:
    job_dir = Path(job["job_dir"])
    while True:
        info = status(job_dir)
        if not info["process_alive"]:
            if not complete(Path(job["run_dir"])):
                raise RuntimeError(f"Job stopped without verified completion: {job['name']}")
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hf-repo-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    for job in jobs(args.root, args.hf_repo_id):
        if complete(Path(job["run_dir"])):
            print(f"SKIP_VERIFIED {job['name']}", flush=True)
            continue
        if status(Path(job["job_dir"]))["process_alive"]:
            raise RuntimeError(f"Already running: {job['name']}")
        env = {**job["env"], "HF_TOKEN": token, "MARIMO_HF_REPO_ID": args.hf_repo_id}
        launch_detached(
            job["command"], cwd=args.root,
            log_path=Path(job["job_dir"]) / "train.log",
            pid_path=Path(job["job_dir"]) / "train.pid",
            state_path=Path(job["job_dir"]) / "state.json",
            env=env,
        )
        wait_for(job, args.poll_seconds)
        print(f"COMPLETE {job['name']}", flush=True)


if __name__ == "__main__":
    main()
