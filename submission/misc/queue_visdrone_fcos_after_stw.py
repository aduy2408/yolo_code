#!/usr/bin/env python3
"""Wait for verified STW-YOLO, then launch VisDrone FCOS jobs sequentially."""
from __future__ import annotations

import os
import subprocess
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/tmp/uv-venv/bin/python"
YOLO_ROOT = Path("/marimo/yolo_code")
MMDET_ROOT = Path("/marimo/mmdet_code")
STW_RUN = YOLO_ROOT / "runs/visdrone_stw"
JOBS = (
    ("fcos_set", "duyle2408/visdrone-fcos-set-runs", "/marimo/mmdet_code/mmdetection/configs/set/fcos_r50_set.py"),
    ("fcos_srtod", "duyle2408/visdrone-fcos-srtod-runs", "/marimo/mmdet_code/SR-TOD/srtod_project/srtod_faster_rcnn/config/srtod-faster-rcnn_r50_fpn_1x_coco.py"),
)


def cli(*args: str, cwd: Path = YOLO_ROOT) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLO_ROOT)
    subprocess.run([PYTHON, "-m", "utils.marimo_ops", *args], cwd=cwd, env=env, check=True)


def read_status(run: Path) -> dict:
    import json
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(YOLO_ROOT)
    result = subprocess.run(
        [PYTHON, "-m", "utils.marimo_ops", "status", "--run-dir", str(run)],
        cwd=YOLO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def launch_job(name: str, repo: str, model_yaml: str) -> None:
    run = Path("/marimo/visdrone_runs") / name
    cli(
        "launch",
        "--cwd", str(MMDET_ROOT),
        "--run-dir", str(run),
        "--artifact-root", str(run),
        "--",
        "/marimo/mmdet-venv/bin/python",
        "/marimo/mmdet_code/train_visdrone_fcos_models_marimo.py",
        "--model", name,
        "--data-root", "/marimo/VisDrone2019",
        "--dataset-root", f"/marimo/visdrone_datasets/{name}",
        "--work-dir", str(run),
        "--epochs", "100",
        "--batch-size", "8",
        "--workers", "8",
        "--seed", "42",
        "--split-seed", "42",
        "--model-yaml", model_yaml,
        "--hf-repo-id", repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/marimo/VisDrone2019")
    parser.add_argument("--model-yaml", default="/marimo/STW-YOLO/Lib/p2_rp5_yolo12s.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--hf-repo-id", default="duyle2408/visdrone-fcos-set-runs")
    parser.parse_known_args()
    while True:
        status = read_status(STW_RUN)
        if status.get("upload_verified") and status.get("required_artifacts", {}).get("results.csv"):
            break
        state = status.get("state") or {}
        if state.get("status") == "exited":
            raise RuntimeError(f"STW did not complete and upload: {state}")
        time.sleep(60)

    for name, repo, model_yaml in JOBS:
        launch_job(name, repo, model_yaml)
        while True:
            status = read_status(Path("/marimo/visdrone_runs") / name)
            if status.get("upload_verified"):
                break
            state = status.get("state") or {}
            if state.get("status") == "exited":
                raise RuntimeError(f"FCOS job failed before verified upload: {name}: {state}")
            time.sleep(60)


if __name__ == "__main__":
    main()
