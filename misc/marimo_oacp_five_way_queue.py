#!/usr/bin/env python3
"""Sequential five-run R2 OACP policy queue.

This queue deliberately holds the detector, split, Mosaic policy, placement,
probability, scale, and base R2 range fixed. Each job changes exactly one
strength/protection knob. It launches the existing verified LEVIR runner and
waits for local plus remote artifacts before starting the next job.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from utils.marimo_ops import launch_detached, status

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python")
POLICIES = ("effect_adaptive", "size_adaptive", "load_adaptive", "curriculum", "hardness_adaptive")
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
    "manifest.json",
)


def _job(args: argparse.Namespace, policy: str, repo: str) -> dict[str, object]:
    project = args.project / f"levir_oacp_r2_{policy}_seed{args.seed}"
    run_dir = project / "p2p3p4_oacp"
    command = [
        PYTHON, "misc/train_context_aug_matrix.py",
        "--data-root", str(args.data_root),
        "--dataset-root", str(args.dataset_root),
        "--project", str(project),
        "--hf-repo-id", repo,
        "--only", "p2p3p4_oacp",
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--device", args.device,
        "--mosaic", "0.0",
        "--close-mosaic", "0",
    ]
    env = {
        "YOLO_CONTEXT_AUG": "oacp",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "OACP_PLACEMENT": "pre_transform",
        "OACP_PROB_POLICY": "fixed",
        "OACP_STRENGTH_POLICY": "fixed" if policy == "size_adaptive" else policy,
        "OACP_PROTECTION_POLICY": "size_adaptive" if policy == "size_adaptive" else "fixed",
        "MARIMO_HF_REPO_ID": repo,
    }
    if policy == "effect_adaptive":
        if args.effect_target <= 0:
            raise ValueError("--effect-target must be calibrated and > 0 for effect_adaptive")
        env.update({"OACP_EFFECT_POLICY": "adaptive", "OACP_TARGET_EFFECT": str(args.effect_target)})
    else:
        env["OACP_EFFECT_POLICY"] = "fixed"
    if policy == "size_adaptive":
        env.update({"OACP_SIZE_EXPAND_MIN": "2.0", "OACP_SIZE_EXPAND_MAX": "4.0", "OACP_SIZE_EXPAND_SMAX": "32.0"})
    return {
        "name": f"levir_r2_{policy}_seed{args.seed}",
        "repo": repo,
        "job_dir": args.project / "jobs" / f"levir_r2_{policy}_seed{args.seed}",
        "run_dir": run_dir,
        "command": command,
        "env": env,
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
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required before launching the adaptive queue")
    if (args.seed, args.split_seed, args.epochs, args.patience) != (42, 42, 100, 0):
        raise ValueError("R2 adaptive matrix requires seed=42, split-seed=42, epochs=100, patience=0")
    repos = {
        "effect_adaptive": args.effect_repo,
        "size_adaptive": args.size_repo,
        "load_adaptive": args.load_repo,
        "curriculum": args.curriculum_repo,
        "hardness_adaptive": args.hardness_repo,
    }
    jobs = [_job(args, policy, repos[policy]) for policy in POLICIES]
    for job in jobs:
        job_dir = Path(job["job_dir"])
        if _complete(job):
            print(f"SKIP_VERIFIED {job['name']}", flush=True)
            continue
        if status(job_dir)["process_alive"]:
            raise RuntimeError(f"Already running: {job['name']}")
        env = {**job["env"], "HF_TOKEN": os.environ["HF_TOKEN"]}
        launch_detached(
            job["command"], cwd=ROOT, log_path=job_dir / "train.log",
            pid_path=job_dir / "train.pid", state_path=job_dir / "state.json", env=env,
        )
        _wait(job, args.poll_seconds)
        print(f"COMPLETE {job['name']} repo={job['repo']}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/marimo/LevirShip/LevirShipData"))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/levir_r2_adaptive_split42")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_r2_adaptive")
    parser.add_argument("--effect-target", type=float, required=True)
    parser.add_argument("--effect-repo", required=True)
    parser.add_argument("--size-repo", required=True)
    parser.add_argument("--load-repo", required=True)
    parser.add_argument("--curriculum-repo", required=True)
    parser.add_argument("--hardness-repo", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
