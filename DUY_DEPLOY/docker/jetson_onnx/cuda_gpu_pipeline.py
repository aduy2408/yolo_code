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
        self._library.launch_yolo_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        self._library.launch_yolo_decode.restype = ctypes.c_int

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

    def decode(
        self,
        prediction: torch.Tensor,
        num_classes: int,
        confidence: float,
        boxes: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        classes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prediction.shape != (1, num_classes + 4, 8400) or not prediction.is_cuda:
            raise ValueError("prediction shape does not match the class count")
        boxes = boxes if boxes is not None else torch.empty((4, 8400), device=prediction.device)
        scores = scores if scores is not None else torch.empty((8400,), device=prediction.device)
        classes = classes if classes is not None else torch.empty((8400,), device=prediction.device, dtype=torch.int32)
        status = self._library.launch_yolo_decode(
            ctypes.c_void_p(prediction.data_ptr()),
            ctypes.c_void_p(boxes.data_ptr()),
            ctypes.c_void_p(scores.data_ptr()),
            ctypes.c_void_p(classes.data_ptr()),
            num_classes,
            confidence,
            ctypes.c_void_p(torch.cuda.current_stream(prediction.device).cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"CUDA decode failed with error code {status}")
        return boxes.transpose(0, 1), scores, classes


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
    """Decode YOLOv8 [1,4+classes,8400] and class-aware NMS without leaving the GPU."""
    if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[1] < 5 or prediction.shape[2] != 8400:
        raise ValueError("prediction must have shape [1,4+classes,8400]")
    if not prediction.is_cuda:
        raise ValueError("prediction must be a CUDA tensor")
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


def gpu_nms_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
    original_width: int,
    original_height: int,
    confidence: float = 0.001,
    nms_iou: float = 0.5,
    max_det: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = scores >= confidence
    boxes = boxes[selected]
    scores = scores[selected]
    classes = classes[selected]
    if not boxes.numel():
        return boxes.reshape(0, 4), scores, classes
    keep = nms(boxes + classes[:, None].float() * 7680.0, scores, nms_iou)[:max_det]
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
        engine_bytes = Path(engine_path).read_bytes()
        payload_start = engine_bytes.find(b"ftrt")
        if payload_start > 0:
            engine_bytes = engine_bytes[payload_start:]
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self.output_name = next(name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        if tuple(self.engine.get_tensor_shape(self.input_name)) != (1, 3, input_size, input_size):
            raise ValueError("TensorRT engine input shape does not match the fixed pipeline")
        output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[1] < 5 or output_shape[2] != 8400:
            raise ValueError("TensorRT engine must expose raw YOLOv8 output [1,4+classes,8400]")
        self.output_shape = output_shape
        self.num_classes = output_shape[1] - 4
        self.context = self.engine.create_execution_context()
        self.input = torch.empty((1, 3, input_size, input_size), device="cuda", dtype=torch.float32)
        self.output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
        self.decoded_boxes = torch.empty((4, 8400), device="cuda", dtype=torch.float32)
        self.decoded_scores = torch.empty((8400,), device="cuda", dtype=torch.float32)
        self.decoded_classes = torch.empty((8400,), device="cuda", dtype=torch.int32)

    def infer(self, bgr: torch.Tensor, confidence: float = 0.001, nms_iou: float = 0.5, max_det: int = 300):
        prepared, scale, pad_x, pad_y = self.preprocessor(bgr, self.input)
        self.context.set_tensor_address(self.input_name, prepared.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(bgr.device).cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        boxes, scores, classes = self.preprocessor.decode(
            self.output,
            self.num_classes,
            confidence,
            self.decoded_boxes,
            self.decoded_scores,
            self.decoded_classes,
        )
        return gpu_nms_postprocess(
            boxes,
            scores,
            classes,
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
        boxes, scores, classes = self.preprocessor.decode(
            self.output,
            self.num_classes,
            confidence,
            self.decoded_boxes,
            self.decoded_scores,
            self.decoded_classes,
        )
        return gpu_nms_postprocess(
            boxes,
            scores,
            classes,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height,
            confidence,
            nms_iou,
            max_det,
        )

    def capture_resized_graph(self, bgr_resized: torch.Tensor, confidence: float = 0.001) -> None:
        """Capture fixed-shape pack, TensorRT, and decode; caller owns the persistent input tensor."""
        if bgr_resized.shape != (self.input_size, self.input_size, 3) or not bgr_resized.is_cuda:
            raise ValueError("CUDA Graph input must be a fixed-size CUDA HWC tensor")
        stream = torch.cuda.current_stream(bgr_resized.device)
        for _ in range(20):
            self.preprocessor.pack_resized(bgr_resized, self.input)
            self.context.set_tensor_address(self.input_name, self.input.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output.data_ptr())
            self.context.execute_async_v3(stream.cuda_stream)
            self.preprocessor.decode(
                self.output,
                self.num_classes,
                confidence,
                self.decoded_boxes,
                self.decoded_scores,
                self.decoded_classes,
            )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.preprocessor.pack_resized(bgr_resized, self.input)
            self.context.set_tensor_address(self.input_name, self.input.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output.data_ptr())
            self.context.execute_async_v3(stream.cuda_stream)
            self.preprocessor.decode(
                self.output,
                self.num_classes,
                confidence,
                self.decoded_boxes,
                self.decoded_scores,
                self.decoded_classes,
            )
        torch.cuda.synchronize()
        self._cuda_graph = graph
        self._cuda_graph_input_ptr = bgr_resized.data_ptr()
        self._cuda_graph_confidence = confidence

    def infer_resized_graph(
        self,
        bgr_resized: torch.Tensor,
        original_width: int,
        original_height: int,
        scale: float,
        pad_x: int = 0,
        pad_y: int = 0,
        nms_iou: float = 0.5,
        max_det: int = 300,
    ):
        if getattr(self, "_cuda_graph", None) is None or bgr_resized.data_ptr() != self._cuda_graph_input_ptr:
            raise RuntimeError("Call capture_resized_graph with this persistent input tensor first")
        self._cuda_graph.replay()
        return gpu_nms_postprocess(
            self.decoded_boxes.transpose(0, 1),
            self.decoded_scores,
            self.decoded_classes,
            scale,
            pad_x,
            pad_y,
            original_width,
            original_height,
            self._cuda_graph_confidence,
            nms_iou,
            max_det,
        )
