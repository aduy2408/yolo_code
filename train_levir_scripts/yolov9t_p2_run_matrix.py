"""Declarative YOLOv9t-P2 LEVIR-Ship eight-run matrix.

This module intentionally contains no training side effects. The Marimo runner can
consume ``RUNS`` after model construction, forward, and smoke-train gates pass.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = ROOT / "models_related/models_config/yolov9/levir"

FTAL = {
    "factorized_tal_target": True,
    "factorized_tal_tau": 0.75,
    "factorized_tal_kappa": 1.5,
    "factorized_tal_lambda": 0.5,
    "factorized_tal_s_max": 32.0,
    "factorized_tal_warmup_start": 5,
    "factorized_tal_warmup_end": 15,
    "factorized_tal_p2_only": True,
}
NO_FTAL = {
    "factorized_tal_target": False,
}

RUNS = {
    "run01_kvca_ftal": {
        "run": 1,
        "label": "YOLOv9t-P2 + KVCA + FTAL",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_kvca.yaml",
        "train_kwargs": FTAL,
    },
    "run02_gap_ftal": {
        "run": 2,
        "label": "YOLOv9t-P2 + GAP + FTAL",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_gap.yaml",
        "train_kwargs": FTAL,
    },
    "run03_kvca_no_ftal": {
        "run": 3,
        "label": "YOLOv9t-P2 + KVCA, no FTAL",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_kvca.yaml",
        "train_kwargs": NO_FTAL,
    },
    "run04_baseline": {
        "run": 4,
        "label": "YOLOv9t-P2 baseline",
        "config": CONFIG_ROOT / "yolov9t_p2_levir.yaml",
        "train_kwargs": NO_FTAL,
    },
    "run05_ftal_only": {
        "run": 5,
        "label": "YOLOv9t-P2 + FTAL only",
        "config": CONFIG_ROOT / "yolov9t_p2_levir.yaml",
        "train_kwargs": FTAL,
    },
    "run06_gap_no_ftal": {
        "run": 6,
        "label": "YOLOv9t-P2 + GAP, no FTAL",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_gap.yaml",
        "train_kwargs": NO_FTAL,
    },
    "run07_repdw5_gap": {
        "run": 7,
        "label": "YOLOv9t-P2 + RepDW5 + GAP",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_repdw5_gap.yaml",
        "train_kwargs": NO_FTAL,
    },
    "run08_repdw5_gap_ftal": {
        "run": 8,
        "label": "YOLOv9t-P2 + RepDW5 + GAP + FTAL",
        "config": CONFIG_ROOT / "yolov9t_p2_levir_repdw5_gap.yaml",
        "train_kwargs": FTAL,
    },
}

PROTOCOL = {
    "dataset": "LEVIR-Ship",
    "split": "fixed 2320/788/788",
    "imgsz": 512,
    "batch": 8,
    "epochs": 100,
    "patience": 0,
    "seed": 42,
    "checkpoint": "best.pt",
    "nms_iou": 0.5,
}


def validate_matrix() -> None:
    """Catch accidental omissions or duplicate run IDs before a real launch."""
    assert set(RUNS) == {
        "run01_kvca_ftal", "run02_gap_ftal", "run03_kvca_no_ftal", "run04_baseline",
        "run05_ftal_only", "run06_gap_no_ftal", "run07_repdw5_gap", "run08_repdw5_gap_ftal",
    }
    assert sorted(spec["run"] for spec in RUNS.values()) == list(range(1, 9))
    assert all(Path(spec["config"]).is_file() for spec in RUNS.values())


if __name__ == "__main__":
    validate_matrix()
    for key, spec in RUNS.items():
        print(f'{spec["run"]}: {key} -> {Path(spec["config"]).name}')
