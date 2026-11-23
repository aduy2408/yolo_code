from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Export a checkpoint directly through Ultralytics TensorRT export.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--imgsz", type=int, required=True)
parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
parser.add_argument("--data", type=Path, help="Calibration dataset YAML required for INT8.")
parser.add_argument("--fraction", type=float, default=1.0)
args = parser.parse_args()

if args.precision == "int8" and args.data is None:
    parser.error("--data is required for --precision int8")

args.output.parent.mkdir(parents=True, exist_ok=True)
temporary_checkpoint = args.output.parent / f"{args.output.stem}.pt"
shutil.copy2(args.checkpoint, temporary_checkpoint)
model = YOLO(str(temporary_checkpoint), task="detect")
exported = model.export(
    format="engine",
    imgsz=args.imgsz,
    batch=1,
    half=args.precision == "fp16",
    int8=args.precision == "int8",
    data=str(args.data) if args.data else None,
    fraction=args.fraction,
    device=0,
    dynamic=False,
    workspace=4,
)
source = Path(exported)
if source.resolve() != args.output.resolve():
    shutil.copy2(source, args.output)
print(f"EXPORTED {args.output} bytes={args.output.stat().st_size}", flush=True)
