#!/usr/bin/env python3
"""Run legacy TPH-YOLOv5 under modern NumPy/PyTorch runtimes."""
from __future__ import annotations

import runpy
import sys

import numpy as np

np.int = int
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
if len(sys.argv) < 2:
    raise SystemExit("usage: run_tph_compat.py val.py [args ...]")
script, *args = sys.argv[1:]
sys.argv = [script, *args]
runpy.run_path(script, run_name="__main__")
