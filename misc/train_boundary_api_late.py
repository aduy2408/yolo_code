import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "models_related" / "ultralytics"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL))
for k in list(sys.modules.keys()):
    if k == "ultralytics" or k.startswith("ultralytics."):
        del sys.modules[k]

from ultralytics import YOLO


MODEL_SCALE = "n"
CONFIG_DIR = ROOT / "models_related" / "models_config" / "yolov8"
DATA_ROOT = Path(os.environ.get("VARROA_DATA_ROOT", "/marimo/data"))
DATASET_OUT_DIR = ROOT / "datasets" / "varroa_yolo"
DATA_PATH = DATASET_OUT_DIR / "varroa.yaml"
TRAIN_PROJECT = ROOT / "runs/detect/yolo_related/runs/train_boundary_api_late"
TEST_PROJECT = ROOT / "runs/detect/yolo_related/runs/test_boundary_api_late"

EXPERIMENTS = (
    {
        "key": "baseline_p2_boundary",
        "config": "yolov8_varroa_compare_baseline_p2_boundary.yaml",
    },
    {
        "key": "baseline_p2_boundary_api_boxgrad",
        "config": "yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad.yaml",
    },
    {
        "key": "baseline_p2_boundary_late",
        "config": "yolov8_varroa_compare_baseline_p2_boundary_late.yaml",
    },
    {
        "key": "baseline_p2_boundary_api_boxgrad_late",
        "config": "yolov8_varroa_compare_baseline_p2_boundary_api_boxgrad_late.yaml",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train early vs late Boundary/API P2 placement comparisons."
    )
    parser.add_argument("--scale", default=MODEL_SCALE, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--only",
        choices=tuple(exp["key"] for exp in EXPERIMENTS),
        default=None,
        help="Train only one experiment instead of the full late-placement comparison.",
    )
    parser.add_argument("--test", action="store_true", help="Evaluate best.pt from runs on the test split.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training.")
    parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics training plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run names.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dataset.py.")
    return parser.parse_args()


def prepare_dataset(skip: bool) -> None:
    if skip:
        return
    from misc.prepare_dataset import prepare_dataset as build_varroa_dataset

    print(f"Preparing dataset from {DATA_ROOT} -> {DATASET_OUT_DIR}")
    build_varroa_dataset(str(DATA_ROOT), str(DATASET_OUT_DIR))


def scaled_yaml(config_name: str, scale: str) -> Path:
    src = CONFIG_DIR / config_name
    dst = CONFIG_DIR / config_name.replace("yolov8_", f"yolov8{scale}_")
    shutil.copy(src, dst)
    return dst


def train_one(exp: dict, args: argparse.Namespace) -> str:
    model_yaml = scaled_yaml(exp["config"], args.scale)
    run_name = model_yaml.stem

    model = YOLO(str(model_yaml))
    model.load(f"yolov8{args.scale}.pt", smart_transfer=True)

    train_kwargs = {
        "data": DATA_PATH,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "patience": args.patience,
        "bbox_iou_loss": "wiou",
        "wiou_monotonous": False,
        "project": str(TRAIN_PROJECT),
        "name": run_name,
        "exist_ok": args.exist_ok,
        "val": not args.no_val,
        "plots": not args.no_plots,
        "boundary_contrast": 0.0,
    }

    print("\n" + "=" * 80)
    print(f"Training {exp['key']}: {run_name}")
    print(f"YAML: {model_yaml}")
    print(f"Patience: {args.patience}")
    print("=" * 80)
    model.train(**train_kwargs)
    return run_name


def evaluate_runs(run_names: list[str], args: argparse.Namespace) -> Path:
    rows = []
    for run_name in run_names:
        best = TRAIN_PROJECT / run_name / "weights" / "best.pt"
        if not best.exists():
            print(f"Missing best.pt, skipping test: {best}")
            continue

        print("\n" + "=" * 80)
        print(f"Testing {run_name}")
        print("=" * 80)
        metrics = YOLO(str(best)).val(
            data=DATA_PATH,
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
        writer = csv.DictWriter(f, fieldnames=["run_name", "mAP50", "mAP50-95", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== TEST SUMMARY =====")
    for row in sorted(rows, key=lambda x: x["mAP50"], reverse=True):
        print(
            f"{row['run_name']:70s} | "
            f"mAP50={row['mAP50']:.4f} | "
            f"mAP50-95={row['mAP50-95']:.4f} | "
            f"P={row['precision']:.4f} | R={row['recall']:.4f}"
        )
    print(f"Saved summary: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    prepare_dataset(args.skip_prepare)
    experiments = [exp for exp in EXPERIMENTS if args.only in (None, exp["key"])]
    run_names = [train_one(exp, args) for exp in experiments]
    if args.test:
        evaluate_runs(run_names, args)


if __name__ == "__main__":
    main()
