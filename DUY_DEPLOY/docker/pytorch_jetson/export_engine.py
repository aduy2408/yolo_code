from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Export a checkpoint directly through Ultralytics TensorRT export.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--imgsz", type=int, required=True)
parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
temporary_checkpoint = args.output.parent / f"{args.output.stem}.pt"
shutil.copy2(args.checkpoint, temporary_checkpoint)
model = YOLO(str(temporary_checkpoint), task="detect")
exported = model.export(
    format="engine",
    imgsz=args.imgsz,
    batch=1,
    half=args.precision == "fp16",
    device=0,
    dynamic=False,
    workspace=4,
)
source = Path(exported)
if source.resolve() != args.output.resolve():
    shutil.copy2(source, args.output)
print(f"EXPORTED {args.output} bytes={args.output.stat().st_size}", flush=True)
