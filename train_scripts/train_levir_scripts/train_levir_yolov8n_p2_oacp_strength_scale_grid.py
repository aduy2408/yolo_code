#!/usr/bin/env python3
"""Run a 3x3 OACP strength/scale grid around the existing R3 result.

The R3 center cell is reused from the existing seed-42 artifact and is not
retrained. The other eight Cartesian combinations are new runs. All runs use
p=0.40, protection_expand=3.0, no Mosaic, split_seed=42, training seed 42,
and NMS IoU=0.50.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc import train_context_aug_matrix as matrix

CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"


# Each value is the random [min, max] interval used by OACP. These are the
# three severity levels already represented by R2, R3, and R5.
STRENGTH_RANGES = (
    (0.10, 0.25),  # R2
    (0.20, 0.40),  # R3
    (0.40, 0.65),  # R5
)
SCALE_RANGES = (
    (0.80, 0.95),  # R2
    (0.65, 0.85),  # R3
    (0.45, 0.70),  # R5
)
R3_CELL = (STRENGTH_RANGES[1], SCALE_RANGES[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=[f"s{s}_x{x}" for s in range(1, 4) for x in range(1, 4)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _grid() -> list[tuple[str, dict[str, object]]]:
    runs = []
    for strength_index, strength in enumerate(STRENGTH_RANGES, start=1):
        for scale_index, scale in enumerate(SCALE_RANGES, start=1):
            runs.append((f"s{strength_index}_x{scale_index}", {
                "p": 0.40,
                "expand": 3.0,
                "strength": strength,
                "scale": scale,
            }))
    return runs


def _configure(name: str, spec: dict[str, object]) -> None:
    os.environ.update({
        "YOLO_CONTEXT_AUG": "oacp",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
        "OACP_VARIANT": "current",
        "OACP_P": str(spec["p"]),
        "OACP_PROTECTED_EXPAND": str(spec["expand"]),
        "OACP_STRENGTH_MIN": str(spec["strength"][0]),
        "OACP_STRENGTH_MAX": str(spec["strength"][1]),
        "OACP_SCALE_MIN": str(spec["scale"][0]),
        "OACP_SCALE_MAX": str(spec["scale"][1]),
        "OACP_SWEEP_LABEL": f"strength_scale_grid:{name}",
    })


def _run_args(args: argparse.Namespace, name: str, repo: str, project: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        project=project,
        hf_repo_id=repo,
        only=[name],
        corrected_api_ablation=False,
        corrected_full_matrix=False,
        reuse_from=None,
        reuse_variants=[],
        seed=args.seed,
        split_seed=args.split_seed,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        mosaic=0.0,
        close_mosaic=0,
        deterministic=args.deterministic,
    )


def main() -> None:
    args = parse_args()
    grid = _grid()
    if args.only:
        selected = set(args.only)
        grid = [(name, spec) for name, spec in grid if name in selected]
    matrix.RUNS = {name: (CONFIG, "oacp", {}) for name, _ in grid}

    project = args.project / f"seed{args.seed}"
    repo = f"{args.hf_repo_prefix}-seed{args.seed}"
    for name, spec in grid:
        if (spec["strength"], spec["scale"]) == R3_CELL:
            print(f"REUSE seed={args.seed} variant={name} existing_R3=true", flush=True)
            continue
        _configure(name, spec)
        matrix.run(_run_args(args, name, repo, project))
        print(f"COMPLETE seed={args.seed} variant={name} repo={repo}", flush=True)


if __name__ == "__main__":
    main()
