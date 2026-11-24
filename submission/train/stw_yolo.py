"""Submission entrypoint for STW-YOLO training."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "misc/train_stw_yolo_marimo.py"), run_name="__main__")
