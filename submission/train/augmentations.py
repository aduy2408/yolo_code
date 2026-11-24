"""Submission entrypoint for the augmentation matrix."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "train_scripts/train_all_augmentation_matrix.py"), run_name="__main__")
