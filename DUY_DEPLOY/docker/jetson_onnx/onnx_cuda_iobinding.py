from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class CudaOnnxRuntime:
    """Run a fixed-shape ONNX model with input/output tensors kept on CUDA."""

    def __init__(self, model: str | Path, device_id: int = 0) -> None:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}")
        self.session = ort.InferenceSession(
            str(model),
            providers=[("CUDAExecutionProvider", {"device_id": device_id})],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA was not selected: {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, images: torch.Tensor) -> list[ort.OrtValue]:
        if images.device.type != "cuda":
            raise ValueError("images must already be a CUDA tensor")
        if images.dtype != torch.float32:
            raise ValueError("images must be float32")
        images = images.contiguous()
        binding = self.session.io_binding()
        binding.bind_input(
            self.input_name,
            "cuda",
            images.device.index or 0,
            np.float32,
            tuple(images.shape),
            images.data_ptr(),
        )
        for name in self.output_names:
            binding.bind_output(name, "cuda", images.device.index or 0)
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return binding.get_outputs()
