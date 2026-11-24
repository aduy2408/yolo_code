"""Submission entrypoint for TPH-YOLO training."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "train_scripts/train_tph_yolov5_multidataset.py"), run_name="__main__")
