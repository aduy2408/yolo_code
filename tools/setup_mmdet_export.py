#!/usr/bin/env python3
"""Install and verify the MMDetection ONNX/TensorRT export environment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_PYTHON = "/marimo/mmdet-venv/bin/python"
DEFAULT_MMDEPLOY = Path("/marimo/mmdeploy")
MMDEPLOY_TAG = "v1.3.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Target Python interpreter.")
    parser.add_argument("--mmdeploy-root", type=Path, default=DEFAULT_MMDEPLOY)
    parser.add_argument("--install", action="store_true", help="Install the pinned export dependencies.")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_mmdeploy_checkout(root: Path) -> None:
    if root.exists():
        if not (root / "tools/deploy.py").is_file():
            raise RuntimeError(f"Existing MMDeploy directory is not a checkout: {root}")
        return
    run(["git", "clone", "--branch", MMDEPLOY_TAG, "--depth", "1", "https://github.com/open-mmlab/mmdeploy.git", str(root)])


def patch_torch_exporter(root: Path) -> None:
    path = root / "mmdeploy/apis/onnx/export.py"
    text = path.read_text(encoding="utf-8")
    if "            dynamo=False,\n" not in text:
        needle = "            export_params=True,\n"
        if needle not in text:
            raise RuntimeError(f"Unsupported MMDeploy exporter source: {path}")
        path.write_text(text.replace(needle, needle + "            dynamo=False,\n", 1), encoding="utf-8")



def install(args: argparse.Namespace) -> None:
    ensure_mmdeploy_checkout(args.mmdeploy_root)
    patch_torch_exporter(args.mmdeploy_root)
    run([args.python, "-m", "pip", "install", "setuptools==77.0.3", "wheel"])
    run([args.python, "-m", "pip", "install", "--no-build-isolation", "-e", str(args.mmdeploy_root)])
    run([
        args.python,
        "-m",
        "pip",
        "install",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
    ])
    run([args.python, "-m", "pip", "install", "--no-deps", "onnxscript==0.5.7"])
    run([
        args.python,
        "-m",
        "pip",
        "install",
        "--extra-index-url",
        "https://pypi.nvidia.com",
        "--no-deps",
        "tensorrt==10.13.3.9",
        "tensorrt_cu13==10.13.3.9",
        "tensorrt_cu13_libs==10.13.3.9",
        "tensorrt_cu13_bindings==10.13.3.9",
    ])


def verify(args: argparse.Namespace) -> None:
    probe = """
import importlib.util
import sys
names = ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmdeploy', 'onnx', 'onnxruntime', 'tensorrt')
print('python:', sys.executable)
print({name: bool(importlib.util.find_spec(name)) for name in names})
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
"""
    run([args.python, "-c", probe])
    if not (args.mmdeploy_root / "tools/deploy.py").is_file():
        raise RuntimeError(f"Missing MMDeploy deploy entrypoint: {args.mmdeploy_root / 'tools/deploy.py'}")
    print("deploy.py:", args.mmdeploy_root / "tools/deploy.py")


def main() -> None:
    args = parse_args()
    if args.install:
        install(args)
    verify(args)


if __name__ == "__main__":
    main()
