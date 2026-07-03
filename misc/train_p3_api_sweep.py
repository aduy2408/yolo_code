"""Sweep API hyperparameters on the P3-only PoolingEdge config.

Configs trained:
  - yolov8_varroa_compare_p3_api_boxgrad_edge_pooling.yaml  (base: partial_forward=True)

Sweep axes (13 experiments total, split across 2 machines):
  MACHINE A (exp 0-6):
    0. base_default     — baseline defaults
    1. rho_0002         — rho=0.002
    2. rho_0010         — rho=0.010
    3. rho_0020         — rho=0.020
    4. w_002            — api_weight=0.02
    5. w_010            — api_weight=0.10
    6. w_020            — api_weight=0.20
  MACHINE B (exp 7-12):
    7. mode_foreground  — target_mode=foreground
    8. mode_boundary    — target_mode=boundary
    9. perbox_norm      — use_per_box_norm=True
   10. fgsm_drop10      — fgsm_dropout=True, drop_rate=0.10
   11. fgsm_drop20      — fgsm_dropout=True, drop_rate=0.20
   12. rho_warmup       — use_rho_warmup=True

Usage:
  # Machine A
  python misc/train_p3_api_sweep.py --machine a --scale n --test
  # Machine B
  python misc/train_p3_api_sweep.py --machine b --scale n --test
  # Single label
  python misc/train_p3_api_sweep.py --only rho_0002
  # All (single machine)
  python misc/train_p3_api_sweep.py --machine all
"""

import argparse
import copy
import csv
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "models_related" / "ultralytics"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_SCALE = "n"
CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
BASE_CONFIG = "yolov8_varroa_compare_p3_api_boxgrad_edge_pooling.yaml"
DATA_PATH = ROOT / "datasets" / "varroa_yolo" / "varroa.yaml"
TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train_p3_api_sweep"
TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_p3_api_sweep"

# ---------------------------------------------------------------------------
# Sweep grid  (each dict patches the AdversarialPerturbationInjection args
# inside the YAML at parse time)
# ---------------------------------------------------------------------------
#
# YAML line format we patch:
#   - [-1, 1, AdversarialPerturbationInjection, [rho, api_weight, target_mode,
#       use_partial_forward, use_rho_warmup, warmup_epochs,
#       use_per_box_norm, use_fgsm_dropout, fgsm_drop_rate]]
#
# We keep use_partial_forward=True always.
#
DEFAULT = dict(
    rho=0.005,
    api_weight=0.05,
    target_mode="boxgrad",
    use_partial_forward=True,
    use_rho_warmup=False,
    warmup_epochs=10,
    use_per_box_norm=False,
    use_fgsm_dropout=False,
    fgsm_drop_rate=0.1,
)

SWEEP = [
    # --- 0. baseline (default values) ---
    dict(label="base_default", **DEFAULT),

    # --- 1. rho sweep ---
    dict(label="rho_0002", **{**DEFAULT, "rho": 0.002}),
    dict(label="rho_0010", **{**DEFAULT, "rho": 0.010}),
    dict(label="rho_0020", **{**DEFAULT, "rho": 0.020}),

    # --- 2. api_weight sweep ---
    dict(label="w_002", **{**DEFAULT, "api_weight": 0.02}),
    dict(label="w_010", **{**DEFAULT, "api_weight": 0.10}),
    dict(label="w_020", **{**DEFAULT, "api_weight": 0.20}),

    # --- 3. target_mode sweep ---
    dict(label="mode_foreground", **{**DEFAULT, "target_mode": "foreground"}),
    dict(label="mode_boundary",   **{**DEFAULT, "target_mode": "boundary"}),

    # --- 4. per_box_norm ---
    dict(label="perbox_norm", **{**DEFAULT, "use_per_box_norm": True}),

    # --- 5. fgsm_dropout ---
    dict(label="fgsm_drop10", **{**DEFAULT, "use_fgsm_dropout": True, "fgsm_drop_rate": 0.10}),
    dict(label="fgsm_drop20", **{**DEFAULT, "use_fgsm_dropout": True, "fgsm_drop_rate": 0.20}),

    # --- 6. rho warmup ---
    dict(label="rho_warmup", **{**DEFAULT, "use_rho_warmup": True, "warmup_epochs": 10}),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_args_str(cfg: dict) -> str:
    """Build the YAML list string for AdversarialPerturbationInjection args."""
    return (
        f"[{cfg['rho']}, {cfg['api_weight']}, {cfg['target_mode']}, "
        f"{str(cfg['use_partial_forward']).lower()}, "
        f"{str(cfg['use_rho_warmup']).lower()}, {cfg['warmup_epochs']}, "
        f"{str(cfg['use_per_box_norm']).lower()}, "
        f"{str(cfg['use_fgsm_dropout']).lower()}, {cfg['fgsm_drop_rate']}]"
    )


_API_PATTERN = re.compile(
    r"(- \[-1, 1, AdversarialPerturbationInjection,\s*)\[.*?\]"
)


def patch_yaml(src: Path, dst: Path, cfg: dict) -> None:
    """Copy src YAML to dst with AdversarialPerturbationInjection args patched."""
    text = src.read_text()
    replacement = r"\g<1>" + _api_args_str(cfg)
    patched = _API_PATTERN.sub(replacement, text)
    if patched == text:
        raise RuntimeError(
            f"Could not find AdversarialPerturbationInjection line in {src}"
        )
    dst.write_text(patched)


def scaled_name(config_name: str, scale: str) -> str:
    return config_name.replace("yolov8_", f"yolov8{scale}_", 1)


def train_one(exp: dict, args: argparse.Namespace) -> str:
    """Patch YAML, build run_name, train and return run_name."""
    label = exp["label"]
    # Build a temporary patched YAML (scale-specific name)
    scale_config_name = scaled_name(BASE_CONFIG, args.scale)
    dst_yaml = CONFIG_DIR / scale_config_name.replace(".yaml", f"_sweep_{label}.yaml")

    src_yaml = CONFIG_DIR / BASE_CONFIG
    patch_yaml(src_yaml, dst_yaml, exp)

    run_name = dst_yaml.stem  # e.g. yolov8n_varroa_..._sweep_rho_0002

    print("\n" + "=" * 80)
    print(f"[SWEEP] label={label}  run={run_name}")
    api_str = _api_args_str(exp)
    print(f"  API args: {api_str}")
    print("=" * 80)

    model = YOLO(str(dst_yaml))
    model.load(f"yolov8{args.scale}.pt", smart_transfer=True)

    model.train(
        data=str(DATA_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
        project=str(TRAIN_PROJECT),
        name=run_name,
        exist_ok=args.exist_ok,
        val=not args.no_val,
        plots=not args.no_plots,
    )

    # Clean up temp YAML
    try:
        dst_yaml.unlink()
    except Exception:
        pass

    return run_name


def evaluate_runs(run_names: list[str], args: argparse.Namespace) -> Path:
    rows = []
    for run_name in run_names:
        best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
        if not best.exists():
            print(f"Missing best.pt, skipping: {best}")
            continue

        print("\n" + "=" * 80)
        print(f"Testing {run_name}")
        print("=" * 80)

        metrics = YOLO(str(best)).val(
            data=str(DATA_PATH),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=0.001,
            iou=0.5,
            project=str(TEST_PROJECT),
            name=run_name,
            exist_ok=True,
        )
        rows.append(
            {
                "run_name": run_name,
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
        )

    TEST_PROJECT.mkdir(parents=True, exist_ok=True)
    summary = TEST_PROJECT / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_name", "mAP50", "mAP50-95", "precision", "recall"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== P3 API SWEEP — TEST SUMMARY =====")
    for row in sorted(rows, key=lambda x: x["mAP50"], reverse=True):
        print(
            f"{row['run_name']:80s} | "
            f"mAP50={row['mAP50']:.4f} | "
            f"mAP50-95={row['mAP50-95']:.4f} | "
            f"P={row['precision']:.4f} | R={row['recall']:.4f}"
        )
    print(f"\nSaved summary: {summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep API hyperparameters for the P3-only PoolingEdge config."
    )
    parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--machine",
        choices=("a", "b", "all"),
        default="all",
        help="Which machine partition to run: 'a'=exp 0-6, 'b'=exp 7-12, 'all'=all 13 (default).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run only experiments whose label matches this string (substring match). Overrides --machine.",
    )
    parser.add_argument("--test", action="store_true", help="Evaluate best.pt on test split after training.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    parser.add_argument("--no-plots", action="store_true", help="Disable training plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run directories.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dataset.py.")
    return parser.parse_args()


DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))


def prepare_data() -> None:
    """Prepare the Varroa YOLO dataset."""
    from misc.prepare_dataset import prepare_dataset

    print(f"Preparing dataset from {DATA_ROOT} -> {DATA_PATH.parent}")
    prepare_dataset(str(DATA_ROOT), str(DATA_PATH.parent))


# Machine split indices
# A: 0..6  (7 exps) — baseline + rho sweep + weight sweep
# B: 7..12 (6 exps) — target_mode + perbox_norm + fgsm + warmup
_MACHINE_SPLIT = 7  # first index of Machine B


def main() -> None:
    args = parse_args()

    if not args.skip_prepare:
        prepare_data()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_PATH}")

    # Priority: --only > --machine
    if args.only:
        experiments = [e for e in SWEEP if args.only in e["label"]]
        if not experiments:
            print(f"[ERROR] No experiments matching --only={args.only!r}")
            print(f"Available labels: {[e['label'] for e in SWEEP]}")
            sys.exit(1)
    elif args.machine == "a":
        experiments = SWEEP[:_MACHINE_SPLIT]
    elif args.machine == "b":
        experiments = SWEEP[_MACHINE_SPLIT:]
    else:
        experiments = SWEEP

    labels = [e['label'] for e in experiments]
    print(f"\n[*] P3-only API sweep | machine={args.machine} | "
          f"{len(experiments)} experiments | epochs={args.epochs} | patience={args.patience}")
    print(f"    Labels: {labels}")

    run_names = [train_one(exp, args) for exp in experiments]

    if args.test:
        evaluate_runs(run_names, args)


if __name__ == "__main__":
    main()
