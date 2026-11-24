"""Submission entrypoint for the TinyPerson training matrix."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "train_scripts/train_all_tinyperson.py"), run_name="__main__")
