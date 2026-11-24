"""Submission entrypoint for STW-YOLO training."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "misc/train_stw_yolo_marimo.py"), run_name="__main__")
