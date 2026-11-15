#!/usr/bin/env python3
"""Offline Ultralytics TensorRT export for the DUY deployment registry.

Run this utility from the repository's `ml2` environment on a target-compatible
machine. The web service never imports or invokes this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if os.getenv("CONDA_DEFAULT_ENV") != "ml2":
        raise SystemExit("Activate the ml2 environment before offline export: source /home/duylearch/miniconda3/bin/activate ml2")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = args.output.parent / f".{args.output.stem}.source.pt"
    shutil.copy2(args.checkpoint, temporary_checkpoint)
    try:
        import ultralytics
        from ultralytics import YOLO
        model = YOLO(str(temporary_checkpoint), task="detect")
        exported = model.export(
            format="engine",
            imgsz=args.imgsz,
            batch=1,
            half=args.precision == "fp16",
            device=args.device,
            dynamic=False,
            workspace=4,
        )
        source = Path(exported)
        if source.resolve() != args.output.resolve():
            shutil.copy2(source, args.output)
            source.unlink(missing_ok=True)
            source.with_suffix(".onnx").unlink(missing_ok=True)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError(f"Export did not produce a non-empty engine: {args.output}")
    from PIL import Image
    engine_model = YOLO(str(args.output), task="detect")
    engine_model.predict(
        source=Image.new("RGB", (args.imgsz, args.imgsz)),
        imgsz=args.imgsz,
        batch=1,
        conf=0.001,
        iou=0.5,
        max_det=300,
        device=args.device,
        half=args.precision == "fp16",
        verbose=False,
        save=False,
    )
    backend = getattr(getattr(engine_model, "predictor", None), "model", None)
    backend_format = str(getattr(backend, "format", "")).lower()
    backend_impl = getattr(backend, "backend", backend)
    backend_name = type(backend_impl).__name__
    binding_fp16 = bool(getattr(backend, "fp16", False))
    if backend_format != "engine" or "tensorrt" not in backend_name.lower():
        raise RuntimeError(f"Exported artifact did not load through Ultralytics TensorRT: format={backend_format}, backend={backend_name}")
    manifest = {
        "artifact_path": str(args.output),
        "sha256": sha256(args.output),
        "precision": args.precision,
        "binding_fp16": binding_fp16,
        "precision_note": "engine loaded and TensorRT backend verified; binding dtype is the runtime-observed precision signal",
        "image_size": args.imgsz,
        "batch_size": 1,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "expected_provider": "ultralytics_tensorrt_engine",
        "runtime_backend": backend_name,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
        },
        "nms_iou": 0.5,
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
