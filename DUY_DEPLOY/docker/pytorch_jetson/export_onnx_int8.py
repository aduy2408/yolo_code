from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Export a YOLO checkpoint to calibrated INT8 ONNX with Ultralytics.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML used for calibration.")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--imgsz", type=int, required=True)
parser.add_argument("--fraction", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--device", default="0")
parser.add_argument("--opset", type=int, default=17)
args = parser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)
temporary_checkpoint = Path("/tmp/export_checkpoint.pt")
shutil.copy2(args.checkpoint, temporary_checkpoint)
model = YOLO(str(temporary_checkpoint))
exported = model.export(
    format="onnx",
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    dynamic=False,
    int8=True,
    data=str(args.data),
    fraction=args.fraction,
)
source = Path(exported)
args.output.write_bytes(source.read_bytes())
print(f"EXPORTED {args.output} bytes={args.output.stat().st_size}", flush=True)
