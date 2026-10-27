#!/usr/bin/env python3
"""Build an INT8 TensorRT engine from preprocessed NPZ calibration batches."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True, help="Official TensorRT SDK root.")
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--workspace-mib", type=int, default=64)
    return parser.parse_args()


class NpzCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches: list[Path], input_name: str, cache: Path) -> None:
        super().__init__()
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            self._mode = "pycuda"
        except ImportError:
            from cuda.bindings import runtime as cuda
            self._mode = "cuda-python"

        self._cuda = cuda
        self._batches = batches
        self._input_name = input_name
        self._cache = cache
        self._index = 0
        first = np.load(batches[0])[input_name]
        self._batch_size = int(first.shape[0])
        if self._mode == "pycuda":
            self._device = cuda.mem_alloc(first.nbytes)
        else:
            error, self._device = cuda.cudaMalloc(first.nbytes)
            if error != cuda.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed: {error}")

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self._index >= len(self._batches):
            return None
        with np.load(self._batches[self._index]) as archive:
            batch = np.ascontiguousarray(archive[self._input_name], dtype=np.float32)
        if batch.shape[0] != self._batch_size:
            raise ValueError("All NPZ calibration batches must have the same batch size")
        if self._mode == "pycuda":
            self._cuda.memcpy_htod(self._device, batch)
        else:
            error, = self._cuda.cudaMemcpy(
                self._device,
                batch.ctypes.data,
                batch.nbytes,
                self._cuda.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            if error != self._cuda.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy failed: {error}")
        self._index += 1
        return [int(self._device)]

    def read_calibration_cache(self) -> bytes | None:
        return self._cache.read_bytes() if self._cache.is_file() else None

    def write_calibration_cache(self, cache: bytes) -> None:
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        self._cache.write_bytes(cache)


def main() -> None:
    args = parse_args()
    batches = sorted(args.calibration_dir.glob("batch_*.npz"))
    if not batches:
        raise FileNotFoundError(f"No batch_*.npz files found in {args.calibration_dir}")
    if not args.plugin.is_file():
        raise FileNotFoundError(f"TensorRT plugin not found: {args.plugin}")

    ctypes.CDLL(str(args.plugin), mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
    logger = trt.Logger(trt.Logger.INFO)
    plugin_registry = trt.get_plugin_registry()
    if not any(creator.name == "TRTBatchedNMS" for creator in plugin_registry.plugin_creator_list):
        raise RuntimeError("TRTBatchedNMS is not registered; load the MMDeploy plugin before building")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {args.onnx}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 0
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 1
    calibrator = NpzCalibrator(batches, args.input_name, args.cache)
    config.int8_calibrator = calibrator

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = tuple(int(value) for value in input_tensor.shape)
    if any(value < 1 for value in shape):
        shape = (calibrator.get_batch_size(), 3, 800, 800)
    profile.set_shape(args.input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")
    args.engine.write_bytes(serialized)
    print(f"engine: {args.engine} ({args.engine.stat().st_size} bytes)")
    print(f"calibration cache: {args.cache} ({args.cache.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
