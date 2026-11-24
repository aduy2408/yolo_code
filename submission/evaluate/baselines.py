"""Submission entrypoint for YOLO baseline evaluation."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "evaluate_test/evaluate_yolo_baseline_matrix.py"), run_name="__main__")
