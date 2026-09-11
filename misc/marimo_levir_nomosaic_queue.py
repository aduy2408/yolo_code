#!/usr/bin/env python3
"""Sequential fixed budget/density LEVIR-Ship queue with Mosaic disabled."""
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


def jobs(root: Path, repos: dict[str, str]) -> list[dict[str, object]]:
    common = [
        PYTHON,
        "misc/train_context_aug_matrix.py",
        "--data-root", "/marimo/LevirShip/LevirShipData",
        "--dataset-root", str(root / "datasets/levir_ship_seed42"),
        "--seed", "42", "--split-seed", "42", "--epochs", "100",
        "--patience", "0", "--imgsz", "512", "--batch-size", "8",
        "--workers", "4", "--device", "cuda", "--mosaic", "0",
        "--close-mosaic", "0", "--only",
    ]
    specs = (
        ("v8_budget", "budget", "p2p3p4_oacp", repos["v8_budget"]),
        ("v8_density", "density", "p2p3p4_oacp", repos["v8_density"]),
        ("v9_budget", "budget", "yolov9t_p2p3p4_oacp", repos["v9_budget"]),
        ("v9_density", "density", "yolov9t_p2p3p4_oacp", repos["v9_density"]),
    )
    result = []
    for name, variant, only, repo in specs:
        project = root / "runs" / f"levir_{name}_nomosaic_seed42"
        result.append({
            "name": name,
            "repo": repo,
            "project": project / only,
            "job_dir": root / "jobs" / f"levir_{name}_nomosaic_seed42",
            "command": [*common, only, "--project", str(project), "--hf-repo-id", repo],
            "env": {"OACP_VARIANT": variant, "YOLO_CONTEXT_AUG": "oacp", "YOLO_LEGACY_DOUBLE_OACP": "0"},
        })
    return result


def complete(job: dict[str, object]) -> bool:
    project = Path(job["project"])
    marker = project / "upload_complete.json"
    if not marker.is_file():
        return False
    try:
        return marker.read_text().find(f'"repo_id": "{job["repo"]}"') >= 0 and all(
            (project / item).is_file() for item in REQUIRED
        )
    except OSError:
        return False


def wait_for(job: dict[str, object], poll_seconds: int) -> None:
    while True:
        info = status(Path(job["job_dir"]))
        if not info["process_alive"]:
            if not complete(job):
                raise RuntimeError(f"Job stopped without verified completion: {job['name']}")
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    for name in ("v8-budget", "v8-density", "v9-budget", "v9-density"):
        parser.add_argument(f"--hf-{name}", dest=name.replace("-", "_"), required=True)
    args = parser.parse_args()
    repos = {"v8_budget": args.v8_budget, "v8_density": args.v8_density, "v9_budget": args.v9_budget, "v9_density": args.v9_density}
    for job in jobs(args.root, repos):
        job_dir = Path(job["job_dir"])
        if complete(job):
            print(f"SKIP_VERIFIED {job['name']}", flush=True)
            continue
        if status(job_dir)["process_alive"]:
            raise RuntimeError(f"Already running: {job['name']}")
        env = {**job["env"], "HF_TOKEN": os.environ["HF_TOKEN"], "MARIMO_HF_REPO_ID": job["repo"]}
        launch_detached(job["command"], cwd=args.root, log_path=job_dir / "train.log", pid_path=job_dir / "train.pid", state_path=job_dir / "state.json", env=env)
        wait_for(job, args.poll_seconds)
        print(f"COMPLETE {job['name']} repo={job['repo']}", flush=True)


if __name__ == "__main__":
    main()
