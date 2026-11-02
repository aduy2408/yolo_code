from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--imgsz", type=int, required=True)
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
temporary_checkpoint = Path("/tmp/export_checkpoint.pt")
shutil.copy2(args.checkpoint, temporary_checkpoint)
model = YOLO(str(temporary_checkpoint))
exported = model.export(
    format="onnx",
    imgsz=args.imgsz,
    batch=1,
    device=0,
    half=True,
    dynamic=False,
    simplify=False,
    opset=17,
)
source = Path(exported)
args.output.write_bytes(source.read_bytes())
print(f"EXPORTED {args.output} bytes={args.output.stat().st_size}", flush=True)
