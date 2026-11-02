from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import onnxruntime as ort


REQUIRED = ("CUDAExecutionProvider",)


def main() -> None:
    providers = ort.get_available_providers()
    missing = [provider for provider in REQUIRED if provider not in providers]
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "onnxruntime": ort.__version__,
        "available_providers": providers,
        "models_root": os.environ.get("MODELS_ROOT", "/models"),
        "datasets_root": os.environ.get("DATASETS_ROOT", "/datasets"),
        "results_root": os.environ.get("RESULTS_ROOT", "/results"),
        "model_files": sorted(
            str(path)
            for root in (Path(os.environ.get("MODELS_ROOT", "/models")),)
            if root.exists()
            for path in root.rglob("*.onnx")
        ),
    }
    print(json.dumps(payload, indent=2), flush=True)
    if platform.machine() != "aarch64":
        raise SystemExit("This image must run on Jetson ARM64 (aarch64).")
    if missing:
        raise SystemExit(f"GPU ONNX Runtime providers missing: {missing}")


if __name__ == "__main__":
    main()
