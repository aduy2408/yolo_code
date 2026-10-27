#!/usr/bin/env python3
"""Reproduce the MMDetection TensorRT plugin and engine build."""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_PYTHON = "/marimo/mmdet-venv/bin/python"
DEFAULT_MMDEPLOY = Path("/marimo/mmdeploy")
DEFAULT_SDK = Path("/marimo/tensorrt-official")
DEFAULT_SDK_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/"
    "10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-13.1.tar.gz"
)
DEFAULT_CUDA = Path("/usr/local/cuda-13.1")
DEFAULT_CUDNN = Path("/marimo/mmdet-venv/lib/python3.11/site-packages/nvidia/cudnn")
DEFAULT_WORK_DIR = Path("/marimo/mmdet_exports/levir_rtmdet_seed43/tensorrt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--mmdeploy-root", type=Path, default=DEFAULT_MMDEPLOY)
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--sdk-url", default=DEFAULT_SDK_URL)
    parser.add_argument("--sdk-archive", type=Path)
    parser.add_argument("--cuda-root", type=Path, default=DEFAULT_CUDA)
    parser.add_argument("--cudnn-root", type=Path, default=DEFAULT_CUDNN)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--download-sdk", action="store_true")
    parser.add_argument("--build-plugins", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--build-workers", type=int, default=2)
    return parser.parse_args()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sdk_ready(root: Path) -> bool:
    return all(
        (root / relative).is_file()
        for relative in ("bin/trtexec", "include/NvInfer.h", "lib/libnvinfer.so")
    )


def obtain_sdk(args: argparse.Namespace) -> None:
    if sdk_ready(args.sdk_root):
        return
    if not args.download_sdk:
        raise FileNotFoundError(
            f"TensorRT SDK is incomplete: {args.sdk_root}; use --download-sdk or provide --sdk-root"
        )
    archive = args.sdk_archive or args.sdk_root.with_suffix(".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        print(f"Downloading {args.sdk_url} -> {archive}", flush=True)
        urlretrieve(args.sdk_url, archive)
    args.sdk_root.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(args.sdk_root)])
    if not sdk_ready(args.sdk_root):
        raise RuntimeError(f"TensorRT SDK extraction did not produce a usable SDK: {args.sdk_root}")


def patch_mmdeploy_sources(root: Path) -> None:
    cuda_cmake = root / "cmake/cuda.cmake"
    text = cuda_cmake.read_text(encoding="utf-8")
    updated = text.replace("set(CMAKE_CUDA_STANDARD 14)", "set(CMAKE_CUDA_STANDARD 17)")
    if updated == text and "set(CMAKE_CUDA_STANDARD 17)" not in text:
        raise RuntimeError(f"Unexpected MMDeploy CUDA source: {cuda_cmake}")
    cuda_cmake.write_text(updated, encoding="utf-8")

    ops_cmake = root / "csrc/mmdeploy/backend_ops/tensorrt/CMakeLists.txt"
    text = ops_cmake.read_text(encoding="utf-8")
    old = "file(GLOB_RECURSE BACKEND_OPS_SRCS *.cpp *.cu)"
    new = "file(GLOB BACKEND_OPS_SRCS batched_nms/*.cpp common_impl/nms/*.cpp common_impl/nms/*.cu)"
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise RuntimeError(f"Unexpected MMDeploy TensorRT source list: {ops_cmake}")
    ops_cmake.write_text(text, encoding="utf-8")


def build_plugins(args: argparse.Namespace, env: dict[str, str]) -> None:
    patch_mmdeploy_sources(args.mmdeploy_root)
    build_dir = args.mmdeploy_root / "build"
    run(
        [
            "cmake",
            "-S",
            str(args.mmdeploy_root),
            "-B",
            str(build_dir),
            "--fresh",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(args.mmdeploy_root.resolve()),
            "-DMMDEPLOY_TARGET_DEVICES=cuda;cpu",
            "-DMMDEPLOY_TARGET_BACKENDS=trt",
            "-DTENSORRT_DIR=" + str(args.sdk_root.resolve()),
            "-DCUDNN_DIR=" + str(args.cudnn_root.resolve()),
            "-DCUDA_TOOLKIT_ROOT_DIR=" + str(args.cuda_root.resolve()),
            "-DCMAKE_CUDA_ARCHITECTURES=120",
        ],
        env=env,
    )
    run(["cmake", "--build", str(build_dir), "--parallel", str(args.build_workers)], env=env)
    run(["cmake", "--install", str(build_dir)], env=env)
    plugin = args.mmdeploy_root / "mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
    if not plugin.is_file():
        raise FileNotFoundError(f"TensorRT plugin was not installed: {plugin}")


def patch_onnx_labels(args: argparse.Namespace, env: dict[str, str]) -> None:
    if not args.onnx:
        raise ValueError("--onnx is required with --build-engine")
    code = """
import onnx
from onnx import TensorProto
model = onnx.load(r'{path}')
for output in model.graph.output:
    if output.name == 'labels':
        output.type.tensor_type.elem_type = TensorProto.INT32
onnx.save(model, r'{path}')
""".format(path=args.onnx.resolve())
    run([args.python, "-c", code], env=env)


def build_engine(args: argparse.Namespace, env: dict[str, str]) -> None:
    patch_onnx_labels(args, env)
    onnx = args.onnx.resolve()
    engine = (args.engine or args.work_dir / "end2end.engine").resolve()
    timing_cache = engine.with_name("timing.cache")
    plugin = args.mmdeploy_root / "mmdeploy/lib/libmmdeploy_tensorrt_ops.so"
    run(
        [
            str(args.sdk_root / "bin/trtexec"),
            "--onnx=" + str(onnx),
            "--staticPlugins=" + str(plugin),
            "--saveEngine=" + str(engine),
            "--fp16",
            "--shapes=input:1x3x800x800",
            "--builderOptimizationLevel=0",
            "--avgTiming=1",
            "--maxTactics=1",
            "--timingCacheFile=" + str(timing_cache),
            "--memPoolSize=workspace:64",
            "--skipInference",
        ],
        cwd=args.mmdeploy_root,
        env=env,
    )
    if not engine.is_file():
        raise FileNotFoundError(f"TensorRT engine was not produced: {engine}")
    print(f"engine: {engine} ({engine.stat().st_size} bytes)")


def main() -> None:
    args = parse_args()
    if not (args.build_plugins or args.build_engine or args.download_sdk):
        raise SystemExit("choose at least one of --download-sdk, --build-plugins, or --build-engine")
    obtain_sdk(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    library_paths = [
        args.sdk_root / "lib",
        args.mmdeploy_root / "mmdeploy/lib",
        args.cuda_root / "targets/x86_64-linux/lib",
        args.cudnn_root / "lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LIBRARY_PATH"] = str(args.cuda_root / "targets/x86_64-linux/lib")
    env["CUDA_HOME"] = str(args.cuda_root)
    env["CUDA_PATH"] = str(args.cuda_root)
    if args.build_plugins:
        build_plugins(args, env)
    if args.build_engine:
        build_engine(args, env)


if __name__ == "__main__":
    main()
