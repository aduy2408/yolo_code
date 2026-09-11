#!/usr/bin/env python3
"""Sequential Marimo queue for the matched YOLOv8 P2/P3/P4 OACP runs.

The supervisor itself is launched detached through ``utils.marimo_ops``. It
waits for each job's PID, verifies local artifacts and upload marker, then
launches exactly one next job through the same shared helper.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from utils.marimo_ops import launch_detached, status

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "evaluation_metrics.json",
)


def job_specs(root: Path, repos: dict[str, str]) -> list[dict[str, object]]:
    levir_data = "/marimo/LevirShip/LevirShipData"
    levir_dataset = root / "datasets/levir_ship_seed42"
    tiny_data = "/marimo/TinyPerson"
    tiny_dataset = root / "datasets/tinyperson_seed42"
    levir_common = [
        "--data-root", levir_data, "--dataset-root", str(levir_dataset),
        "--seed", "42", "--epochs", "100", "--patience", "0",
        "--imgsz", "512", "--batch-size", "8", "--workers", "8",
        "--device", "cuda", "--mosaic", "1.0", "--close-mosaic", "10",
        "--only", "p2p3p4_oacp",
    ]
    tiny_common = [
        "--data-root", tiny_data, "--dataset-root", str(tiny_dataset),
        "--seeds", "42", "--split-seed", "42", "--epochs", "100",
        "--patience", "0", "--imgsz", "640", "--batch-size", "8",
        "--workers", "8", "--device", "cuda", "--amp",
        "--confirm-settings",
    ]
    jobs = []
    for dataset, variant, hf_repo, common, runner, project in [
        ("levir", "budget", repos["levir_v8_budget"], levir_common, "misc/train_context_aug_matrix.py", root / "runs/levir_yolov8n_p2p3p4_budget_mosaic_seed42"),
        ("levir", "density", repos["levir_v8_density"], levir_common, "misc/train_context_aug_matrix.py", root / "runs/levir_yolov8n_p2p3p4_density_mosaic_seed42"),
    ]:
        jobs.append({
            "name": f"{dataset}_{variant}", "variant": variant,
            "job_dir": root / "jobs" / f"{dataset}_{variant}_seed42",
            "project": project / "p2p3p4_oacp",
            "env": {"OACP_VARIANT": variant, "YOLO_CONTEXT_AUG": "oacp"},
            "command": [str(root / ".venv/bin/python")] if False else [
                os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python"), runner,
                *common, "--project", str(project), "--hf-repo-id", hf_repo,
            ],
        })
    for variant in ("budget", "density"):
        project = root / f"runs/tinyperson_yolov8n_p2p3p4_{variant}_mosaic_seed42"
        jobs.append({
            "name": f"tinyperson_{variant}", "variant": variant,
            "job_dir": root / "jobs" / f"tinyperson_{variant}_seed42",
            "project": project / f"{variant}_oacp_mosaic" / "seed_42_corner_sw640_sh512",
            "env": {"OACP_VARIANT": variant, "YOLO_CONTEXT_AUG": "oacp"},
            "command": [
                os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python"),
                "train_all_tinyperson_yolo11_oacp_mosaic_matrix.py", *tiny_common,
                "--model", "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3p4_plain.yaml",
                "--project", str(project), "--hf-repo-id", repos["tinyperson_v8"],
                "--variants", f"{variant}_oacp_mosaic",
            ],
        })
    for dataset, variant, hf_repo, common, runner, project, model in [
        ("levir", "budget", repos["levir_v9_budget"], levir_common, "misc/train_context_aug_matrix.py", root / "runs/levir_yolov9t_p2p3p4_budget_mosaic_seed42", "yolov9t_p2p3p4_oacp"),
        ("levir", "density", repos["levir_v9_density"], levir_common, "misc/train_context_aug_matrix.py", root / "runs/levir_yolov9t_p2p3p4_density_mosaic_seed42", "yolov9t_p2p3p4_oacp"),
    ]:
        jobs.append({
            "name": f"{dataset}_yolov9t_{variant}", "variant": variant,
            "job_dir": root / "jobs" / f"{dataset}_yolov9t_{variant}_seed42",
            "project": project / model,
            "env": {"OACP_VARIANT": variant, "YOLO_CONTEXT_AUG": "oacp"},
            "command": [
                os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python"), runner,
                *common, "--project", str(project), "--hf-repo-id", hf_repo,
                "--only", model,
            ],
        })
    for variant in ("budget", "density"):
        project = root / f"runs/tinyperson_yolov9t_p2p3p4_{variant}_mosaic_seed42"
        jobs.append({
            "name": f"tinyperson_yolov9t_{variant}", "variant": variant,
            "job_dir": root / "jobs" / f"tinyperson_yolov9t_{variant}_seed42",
            "project": project / f"{variant}_oacp_mosaic" / "seed_42_corner_sw640_sh512",
            "env": {"OACP_VARIANT": variant, "YOLO_CONTEXT_AUG": "oacp"},
            "command": [
                os.environ.get("MARIMO_PYTHON", "/tmp/uv-venv/bin/python"),
                "train_all_tinyperson_yolo11_oacp_mosaic_matrix.py", *tiny_common,
                "--model", "models_related/models_config/yolov9/tinyperson/yolov9t_tinyperson_p2p3p4_plain.yaml",
                "--project", str(project), "--hf-repo-id", repos["tinyperson_v9"],
                "--variants", f"{variant}_oacp_mosaic",
            ],
        })
    return jobs


def complete(job: dict[str, object]) -> bool:
    project = Path(job["project"])
    return all((project / item).is_file() for item in REQUIRED) and (project / "upload_complete.json").is_file()


def wait_for(job: dict[str, object], interval: int) -> None:
    job_dir = Path(job["job_dir"])
    while True:
        info = status(job_dir)
        if not info["process_alive"]:
            if not complete(job):
                raise RuntimeError(f"Job stopped without verified completion: {job['name']}")
            return
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-job", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    for name in ("levir-v8-budget", "levir-v8-density", "tinyperson-v8", "levir-v9-budget", "levir-v9-density", "tinyperson-v9"):
        parser.add_argument(f"--hf-{name}", dest=name.replace("-", "_"), required=True)
    args = parser.parse_args()
    repos = {
        "levir_v8_budget": args.levir_v8_budget, "levir_v8_density": args.levir_v8_density,
        "tinyperson_v8": args.tinyperson_v8, "levir_v9_budget": args.levir_v9_budget,
        "levir_v9_density": args.levir_v9_density, "tinyperson_v9": args.tinyperson_v9,
    }
    jobs = job_specs(ROOT, repos)
    start = next(i for i, job in enumerate(jobs) if Path(job["job_dir"]).resolve() == args.watch_job.resolve())
    watch_job = args.watch_job.resolve()
    for job in jobs[start:]:
        job_dir = Path(job["job_dir"]).resolve()
        if job_dir != watch_job and not complete(job):
            repo_key = {
                "levir_budget": "levir_v8_budget", "levir_density": "levir_v8_density",
                "tinyperson_budget": "tinyperson_v8", "tinyperson_density": "tinyperson_v8",
                "levir_yolov9t_budget": "levir_v9_budget", "levir_yolov9t_density": "levir_v9_density",
                "tinyperson_yolov9t_budget": "tinyperson_v9", "tinyperson_yolov9t_density": "tinyperson_v9",
            }[job["name"]]
            repo_id = repos[repo_key]
            env = {**job["env"], "HF_TOKEN": os.environ["HF_TOKEN"], "MARIMO_HF_REPO_ID": repo_id}
            job_dir = Path(job["job_dir"])
            launch_detached(job["command"], cwd=ROOT, log_path=job_dir / "train.log", pid_path=job_dir / "train.pid", state_path=job_dir / "state.json", env=env)
        wait_for(job, args.poll_seconds)
        (ROOT / "jobs" / "oacp_queue_state.json").write_text(json.dumps({"completed": job["name"], "verified": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
