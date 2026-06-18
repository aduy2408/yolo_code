#!/usr/bin/env python3
"""Smoke test for the patched local Ultralytics package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Return this repo root in local runs or the mounted Marimo path."""
    candidates = [
        Path(__file__).resolve().parent,
        Path("/marimo/yolo_code").resolve(),
    ]
    for root in candidates:
        if (root / "models_related" / "ultralytics" / "ultralytics" / "__init__.py").is_file():
            return root
    raise FileNotFoundError("Could not find models_related/ultralytics/ultralytics/__init__.py")


def prefer_local_ultralytics(repo_root: Path) -> None:
    """Put the patched local Ultralytics package before site-packages."""
    sys.path[:0] = [
        str(repo_root / "models_related" / "ultralytics"),
        str(repo_root),
    ]


def main() -> None:
    repo_root = find_repo_root()
    prefer_local_ultralytics(repo_root)

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT

    print(ultralytics.__file__)
    print("bbox_iou_loss", "bbox_iou_loss" in DEFAULT_CFG_DICT)
    print("wiou_monotonous", "wiou_monotonous" in DEFAULT_CFG_DICT)
    print("YOLO", YOLO)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "prepare_dataset.py"),
            "--root",
            "/marimo/data",
            "--out-dir",
            "datasets/varroa_yolo",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
