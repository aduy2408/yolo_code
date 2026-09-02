"""Regression tests for the historical custom model YAML corpus.

These tests intentionally use the legacy fork only as a compatibility oracle. They do
not import or mutate the upstream submodule in-process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
YAML_ROOT = ROOT / "models_related" / "models_config"
LEGACY_PACKAGE = ROOT / "models_related" / "ultralytics"


def _yaml_files() -> list[Path]:
    return sorted(YAML_ROOT.rglob("*.yaml"))


def _module_names(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text())
    names = set()
    for section in ("backbone", "head"):
        for row in data.get(section, []) or []:
            if isinstance(row, list) and len(row) >= 3 and isinstance(row[2], str):
                names.add(row[2])
    return names


def _run_legacy_yaml(path: Path) -> subprocess.CompletedProcess[str]:
    script = """
from ultralytics import YOLO
import sys
model = YOLO(sys.argv[1], task='detect')
assert len(model.model.model) > 0
assert sum(p.numel() for p in model.model.parameters()) > 0
print('PASS')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LEGACY_PACKAGE)
    return subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def test_historical_yaml_corpus_is_valid_and_nonempty() -> None:
    files = _yaml_files()
    model_files = []
    for path in files:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), path
        if data.get("backbone") and data.get("head"):
            model_files.append(path)
            assert _module_names(path), path
    assert len(model_files) >= 200


@pytest.mark.parametrize(
    "relative_path",
    [
        "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_plain.yaml",
        "models_related/models_config/yolov8/kvca_sweep/yolov8_kvca_a_sc_avg_p2sr4_p3sr4.yaml",
        "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_context.yaml",
    ],
)
def test_representative_custom_yamls_load_on_legacy_oracle(relative_path: str) -> None:
    result = _run_legacy_yaml(ROOT / relative_path)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "PASS" in result.stdout


def test_clean_upstream_rejects_unmigrated_legacy_symbol() -> None:
    path = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_surgical_a_p3_context.yaml"
    script = """
from ultralytics import YOLO
import sys
YOLO(sys.argv[1], task='detect')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "vendor" / "ultralytics_upstream")
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode != 0
    error = result.stdout + result.stderr
    assert "KVCompressedAttention" in error or "RepC2f" in error
