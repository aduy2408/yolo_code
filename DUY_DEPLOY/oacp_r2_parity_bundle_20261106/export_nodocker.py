#!/usr/bin/env python3
"""Export parity checkpoints to TensorRT engines with Ultralytics, without Docker."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
CHECKPOINTS = BUNDLE / "checkpoints"
ENGINES = BUNDLE / "output" / "engines"
SIZES = {"tinyperson": 640, "levirship": 512, "varroa": 640}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*SIZES, "all"), default="all")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    datasets = tuple(SIZES) if args.dataset == "all" else (args.dataset,)
    precisions = ("fp32", "fp16") if args.precision == "both" else (args.precision,)
    checkpoints = [
        path
        for dataset in datasets
        for path in sorted((CHECKPOINTS / dataset).glob("*/seed*/best.pt"))
    ]
    if not checkpoints:
        raise SystemExit(f"No checkpoints under {CHECKPOINTS}")

    for checkpoint in checkpoints:
        dataset, model, seed = checkpoint.relative_to(CHECKPOINTS).parts[:3]
        for precision in precisions:
            output = ENGINES / dataset / model / seed / f"{precision}.engine"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                print(f"EXISTS {output}", flush=True)
                continue
            print(f"EXPORT {dataset}/{model}/{seed} {precision}", flush=True)
            loaded = YOLO(str(checkpoint), task="detect")
            exported = Path(
                loaded.export(
                    format="engine",
                    imgsz=SIZES[dataset],
                    batch=1,
                    half=precision == "fp16",
                    dynamic=False,
                    device=args.device,
                    workspace=4,
                    verbose=False,
                )
            )
            if exported.resolve() != output.resolve():
                shutil.copy2(exported, output)
            print(f"EXPORTED {output}", flush=True)


if __name__ == "__main__":
    main()
