#!/usr/bin/env python3
"""Queue M4 after the M1/M2/M3 runner verifies M3 upload."""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "runs/oaief_modulation/M3/upload_complete.json"
SHA = "c981750253d20baad8804f59c7c9c4832ae4c78e"
while not MARKER.is_file():
    time.sleep(30)
subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
subprocess.run(["git", "checkout", "--detach", SHA], cwd=ROOT, check=True)
command = [
    sys.executable, "-m", "utils.marimo_ops", "launch",
    "--cwd", str(ROOT), "--run-dir", str(ROOT / "runs/oaief_global_object"), "--",
    sys.executable, "misc/train_oaief_global_object.py",
    "--data-root", "/marimo/LevirShip/LevirShipData",
    "--dataset-root", str(ROOT / "datasets"),
    "--project", str(ROOT / "runs/oaief_global_object"),
    "--hf-repo-id", "duyle2408/object-aware-hit-yolov8n-p2",
    "--seed", "42", "--epochs", "100", "--patience", "0", "--imgsz", "512",
    "--batch-size", "8", "--device", "0", "--workers", "4", "--amp",
]
subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
