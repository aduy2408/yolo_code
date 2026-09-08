#!/usr/bin/env python3
"""Run the seven-run aggressive OACP LEVIR sweep.

R1-R5 screen severity at the fixed reference protection radius.  R6-R7 reuse
the best R1-R5 validation configuration at narrow and wide protection radii.
The current reference run is intentionally not rerun.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from misc import train_context_aug_matrix as matrix

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"

SCREEN = {
    "R1_sparse_mild": {"p": 0.10, "expand": 3.0, "strength": (0.10, 0.25), "scale": (0.80, 0.95)},
    "R2_frequent_mild": {"p": 0.40, "expand": 3.0, "strength": (0.10, 0.25), "scale": (0.80, 0.95)},
    "R3_frequent_current": {"p": 0.40, "expand": 3.0, "strength": (0.20, 0.40), "scale": (0.65, 0.85)},
    "R4_strong": {"p": 0.20, "expand": 3.0, "strength": (0.40, 0.65), "scale": (0.45, 0.70)},
    "R5_very_strong_stress": {"p": 0.40, "expand": 3.0, "strength": (0.40, 0.65), "scale": (0.45, 0.70)},
}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hf-repo-id", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _run_args(a: argparse.Namespace, name: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=a.data_root, dataset_root=a.dataset_root, project=a.project,
        hf_repo_id=a.hf_repo_id, only=[name], corrected_api_ablation=False,
        corrected_full_matrix=False, reuse_from=None, reuse_variants=[],
        seed=a.seed, split_seed=a.split_seed, epochs=a.epochs, patience=a.patience,
        imgsz=a.imgsz, batch_size=a.batch_size, device=a.device,
        workers=a.workers, amp=a.amp,
        mosaic=0.0, close_mosaic=0,
    )


def _configure(name: str, spec: dict[str, object], selected_from: str = "") -> None:
    os.environ.update({
        "OACP_P": str(spec["p"]),
        "OACP_PROTECTED_EXPAND": str(spec["expand"]),
        "OACP_STRENGTH_MIN": str(spec["strength"][0]),
        "OACP_STRENGTH_MAX": str(spec["strength"][1]),
        "OACP_SCALE_MIN": str(spec["scale"][0]),
        "OACP_SCALE_MAX": str(spec["scale"][1]),
        "OACP_SWEEP_LABEL": f"aggressive_oacp:{name}:selected_from={selected_from}",
    })


def main() -> None:
    a = _args()
    matrix.RUNS = {name: (CONFIG, "oacp", {}) for name in (*SCREEN, "R6_narrow_protection", "R7_wide_protection")}
    matrix.require_training_context(hf_repo_id=a.hf_repo_id)

    for name, spec in SCREEN.items():
        _configure(name, spec)
        matrix.run(_run_args(a, name))

    scores = {}
    for name in SCREEN:
        metrics = json.loads((a.project / name / "evaluation_metrics.json").read_text())
        scores[name] = metrics["val/metrics/mAP50(B)"]
    best_name = max(scores, key=scores.get)
    best = dict(SCREEN[best_name])
    for name, expand in (("R6_narrow_protection", 1.5), ("R7_wide_protection", 5.0)):
        spec = dict(best)
        spec["expand"] = expand
        _configure(name, spec, selected_from=best_name)
        matrix.run(_run_args(a, name))
    print(json.dumps({"screen_val_mAP50": scores, "selected_best_severity": best_name}, indent=2), flush=True)


if __name__ == "__main__":
    main()
