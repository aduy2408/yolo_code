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
    parser.add_argument("--build-plugins", action="store_true", help="Build and verify MMDeploy TensorRT custom ops.")
    parser.add_argument("--tensorrt-sdk", type=Path, help="TensorRT SDK root containing include/ and lib/.")
    parser.add_argument("--cudnn-dir", type=Path, help="Optional cuDNN root passed to CMake.")
    parser.add_argument("--build-dir", type=Path, help="CMake build directory; defaults to <mmdeploy-root>/build.")
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



def build_plugins(args: argparse.Namespace) -> None:
    if not args.tensorrt_sdk:
        raise ValueError("--tensorrt-sdk is required with --build-plugins")
    sdk = args.tensorrt_sdk.resolve()
    if not (sdk / "include/NvInfer.h").is_file() or not (sdk / "lib/libnvinfer.so").exists():
        raise FileNotFoundError(f"TensorRT SDK must contain include/NvInfer.h and lib/libnvinfer.so: {sdk}")
    build_dir = (args.build_dir or args.mmdeploy_root / "build").resolve()
    configure = [
        "cmake",
        "-S",
        str(args.mmdeploy_root),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(args.mmdeploy_root.resolve()),
        "-DMMDEPLOY_TARGET_DEVICES=cuda;cpu",
        "-DMMDEPLOY_TARGET_BACKENDS=trt",
        "-DBUILD_TENSORRT_OPS=ON",
        "-DTENSORRT_DIR=" + str(sdk),
    ]
    if args.cudnn_dir:
        configure.append("-DCUDNN_DIR=" + str(args.cudnn_dir.resolve()))
    run(configure)
    run(["cmake", "--build", str(build_dir), "--parallel", "4"])
    run(["cmake", "--install", str(build_dir)])


def verify_plugins(args: argparse.Namespace) -> None:
    library = args.mmdeploy_root / "lib/libmmdeploy_tensorrt_ops.so"
    if not library.is_file():
        raise FileNotFoundError(f"MMDeploy TensorRT plugin library not found: {library}")
    probe = """
import tensorrt as trt
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
assert load_tensorrt_plugin()
names = [creator.name for creator in trt.get_plugin_registry().plugin_creator_list]
assert 'TRTBatchedNMS' in names, names
print('TRTBatchedNMS: available')
"""
    run([args.python, "-c", probe], cwd=args.mmdeploy_root)


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
    if args.build_plugins:
        build_plugins(args)
    verify(args)
    if args.build_plugins:
        verify_plugins(args)


if __name__ == "__main__":
    main()
