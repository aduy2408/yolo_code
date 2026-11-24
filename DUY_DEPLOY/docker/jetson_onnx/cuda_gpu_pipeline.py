from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import torch
from torchvision.ops import nms

from ultralytics.utils.ops import xywh2xyxy


CUDA_SOURCE = Path(__file__).with_name("cuda_preprocess.cu")
DEFAULT_LIBRARY = CUDA_SOURCE.with_suffix(".so")


class CudaPreprocessor:
    """Persistent CUDA letterbox + BGR-to-RGB + normalization for a fixed YOLO input."""

    def __init__(self, output_size: int = 640, library: Path = DEFAULT_LIBRARY) -> None:
        self.output_size = output_size
        self.library = Path(library)
        if not self.library.is_file():
            raise FileNotFoundError(f"Missing CUDA preprocessing library: {self.library}; run build_cuda_pipeline().")
        self._library = ctypes.CDLL(str(self.library))
        self._library.launch_yolo_pack.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.launch_yolo_pack.restype = ctypes.c_int
        self._library.launch_yolo_preprocess.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.launch_yolo_preprocess.restype = ctypes.c_int

    def _output(self, bgr: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != torch.uint8 or not bgr.is_cuda:
            raise ValueError("bgr must be a CUDA uint8 HWC tensor")
        if output is None:
            output = torch.empty(
                (1, 3, self.output_size, self.output_size), device=bgr.device, dtype=torch.float32
            )
        if output.shape != (1, 3, self.output_size, self.output_size) or output.dtype != torch.float32:
            raise ValueError("output must be a persistent FLOAT NCHW tensor with the configured size")
        return output

    def __call__(self, bgr: torch.Tensor, output: torch.Tensor | None = None) -> tuple[torch.Tensor, float, int, int]:
        bgr = bgr.contiguous()
        output = self._output(bgr, output)
        height, width = bgr.shape[:2]
        scale = min(self.output_size / width, self.output_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        pad_x = (self.output_size - resized_width) // 2
        pad_y = (self.output_size - resized_height) // 2
        status = self._library.launch_yolo_preprocess(
            ctypes.c_void_p(bgr.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            width,
            height,
            self.output_size,
            ctypes.c_void_p(torch.cuda.current_stream(bgr.device).cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"CUDA preprocess failed with error code {status}")
        return output, scale, pad_x, pad_y
    def pack_resized(self, bgr: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Pack an already-resized BGR image exactly, leaving resize semantics to OpenCV."""
        bgr = bgr.contiguous()
        output = self._output(bgr, output)
        height, width = bgr.shape[:2]
        if (height, width) != (self.output_size, self.output_size):
            raise ValueError("pack_resized requires an image with the configured square size")
        status = self._library.launch_yolo_pack(
            ctypes.c_void_p(bgr.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            width,
            height,
            ctypes.c_void_p(torch.cuda.current_stream(bgr.device).cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"CUDA packing failed with error code {status}")
        return output


def build_cuda_pipeline(output: Path = DEFAULT_LIBRARY, arch: str | None = None) -> Path:
    output = Path(output)
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "nvcc",
            "-O3",
            f"-arch={arch}",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(CUDA_SOURCE),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def gpu_postprocess(
    prediction: torch.Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_width: int,
    original_height: int,
    confidence: float = 0.001,
    nms_iou: float = 0.5,
    max_det: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode YOLOv8 [1,84,8400] and class-aware NMS without leaving the GPU."""
    if prediction.shape != (1, 84, 8400) or not prediction.is_cuda:
        raise ValueError("prediction must be a CUDA tensor with shape [1,84,8400]")
    values = prediction[0].transpose(0, 1)
    boxes = xywh2xyxy(values[:, :4])
    scores, classes = values[:, 4:].max(dim=1)
    selected = scores >= confidence
    boxes = boxes[selected]
    scores = scores[selected]
    classes = classes[selected]
    if not boxes.numel():
        return boxes.reshape(0, 4), scores, classes
    keep = nms(boxes + classes[:, None] * 7680.0, scores, nms_iou)[:max_det]
    boxes = boxes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_height)
    return boxes, scores[keep], classes[keep]


class TensorRTGpuPipeline:
    """Persistent TensorRT execution with CUDA preprocess and GPU-only postprocess."""

    def __init__(self, engine_path: Path, input_size: int = 640) -> None:
        import tensorrt as trt

        self.input_size = input_size
        self.preprocessor = CudaPreprocessor(input_size)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        if tuple(self.engine.get_tensor_shape(self.input_name)) != (1, 3, input_size, input_size):
            raise ValueError("TensorRT engine input shape does not match the fixed pipeline")
        if tuple(self.engine.get_tensor_shape(self.output_name)) != (1, 84, 8400):
            raise ValueError("TensorRT engine must expose raw YOLOv8 output [1,84,8400]")
        self.context = self.engine.create_execution_context()
        self.input = torch.empty((1, 3, input_size, input_size), device="cuda", dtype=torch.float32)
        self.output = torch.empty((1, 84, 8400), device="cuda", dtype=torch.float32)

    def infer(self, bgr: torch.Tensor, confidence: float = 0.001, nms_iou: float = 0.5, max_det: int = 300):
        prepared, scale, pad_x, pad_y = self.preprocessor(bgr, self.input)
        self.context.set_tensor_address(self.input_name, prepared.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(bgr.device).cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return gpu_postprocess(
            self.output,
            scale,
            pad_x,
            pad_y,
            int(bgr.shape[1]),
            int(bgr.shape[0]),
            confidence,
            nms_iou,
            max_det,
        )

    def infer_resized(
        self,
        bgr_resized: torch.Tensor,
        original_width: int,
        original_height: int,
        scale: float,
        pad_x: int = 0,
        pad_y: int = 0,
        confidence: float = 0.001,
        nms_iou: float = 0.5,
        max_det: int = 300,
    ):
        """Exact-compatible path when resize/letterbox was performed by the reference OpenCV pipeline."""
        prepared = self.preprocessor.pack_resized(bgr_resized, self.input)
        self.context.set_tensor_address(self.input_name, prepared.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(bgr_resized.device).cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return gpu_postprocess(
            self.output,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height,
            confidence,
            nms_iou,
            max_det,
        )
