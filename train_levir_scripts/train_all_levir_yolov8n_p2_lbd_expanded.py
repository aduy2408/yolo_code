#!/usr/bin/env python3
"""Train/evaluate/upload Conv48 versus expanded LBD48 controls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_all_levir_yolov8n_p2_local_basis as base

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
base.EXPERIMENT = "levir_yolov8n_p2_lbd_expanded"
base.HF_REPO = "duyle2408/levir-yolov8n-p2-local-basis"
base.VARIANTS = {
    "conv48": CONFIG_ROOT / "yolov8n_p2_fpn_only_conv48.yaml",
    "lbd48_fixed": CONFIG_ROOT / "yolov8n_p2_fpn_only_lbd48_fixed.yaml",
    "lbd48_adaptive": CONFIG_ROOT / "yolov8n_p2_fpn_only_lbd48_adaptive.yaml",
}
base.workflow.EXPERIMENT = base.EXPERIMENT
base.workflow.HF_REPO = base.HF_REPO
base.workflow.VARIANTS = base.VARIANTS


if __name__ == "__main__":
    base.main()
