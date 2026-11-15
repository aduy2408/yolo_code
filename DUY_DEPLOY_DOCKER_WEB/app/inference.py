from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image, ImageDraw

from .config import Settings
from .output_store import OutputStore
from .registry import ModelArtifact, ModelRegistry, RegistryError, verify_artifact
from .telemetry import RequestRecord, Telemetry


@dataclass
class LoadedModel:
    artifact: ModelArtifact
    model: Any
    load_ms: float
    warmup_ms: float
    backend_identity: str


class ModelCache:
    def __init__(self, settings: Settings, registry: ModelRegistry):
        self.settings = settings
        self.registry = registry
        self.loaded: dict[tuple[str, str], LoadedModel] = {}
        self.errors: dict[tuple[str, str], str] = {}
        self.lock = Lock()
        self.startup: dict[str, Any] = {"status": "not_started", "models": {}}

    def preload(self) -> None:
        started = time.perf_counter_ns()
        self.startup = {"status": "loading", "models": {}}
        if not self.settings.preload_models:
            self.startup.update({"status": "deferred", "total_ms": 0.0})
            return
        for artifact in self.registry.artifacts.values():
            try:
                self._load(artifact)
            except Exception as exc:  # startup must expose a readiness error, not hide it
                self.errors[artifact.key] = f"{type(exc).__name__}: {exc}"
                self.startup["models"][f"{artifact.model_id}:{artifact.precision}"] = {"status": "error", "error": type(exc).__name__}
        all_loaded = bool(self.registry.artifacts) and len(self.loaded) == len(self.registry.artifacts) and not self.errors
        self.startup["status"] = "ready" if all_loaded else ("degraded" if self.loaded else "empty")
        self.startup["total_ms"] = round((time.perf_counter_ns() - started) / 1_000_000, 4)

    def _load(self, artifact: ModelArtifact) -> LoadedModel:
        with self.lock:
            if artifact.key in self.loaded:
                return self.loaded[artifact.key]
            verify_artifact(artifact)
            started = time.perf_counter_ns()
            from ultralytics import YOLO
            model = YOLO(str(artifact.artifact_path), task="detect")
            load_ms = (time.perf_counter_ns() - started) / 1_000_000
            warmup_started = time.perf_counter_ns()
            from PIL import Image
            model.predict(
                source=Image.new("RGB", (artifact.image_size, artifact.image_size)),
                imgsz=artifact.image_size,
                batch=1,
                conf=artifact.conf,
                iou=artifact.nms_iou,
                max_det=artifact.max_det,
                device=self.settings.device,
                half=artifact.precision == "fp16",
                verbose=False,
                save=False,
            )
            warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
            backend_identity = assert_tensorrt_backend(model, artifact.expected_provider)
            loaded = LoadedModel(artifact, model, load_ms, warmup_ms, backend_identity)
            self.loaded[artifact.key] = loaded
            self.startup["models"][f"{artifact.model_id}:{artifact.precision}"] = {
                "status": "loaded",
                "load_ms": round(load_ms, 4),
                "warmup_ms": round(warmup_ms, 4),
                "backend": backend_identity,
            }
            return loaded

    def get(self, artifact: ModelArtifact) -> LoadedModel:
        loaded = self.loaded.get(artifact.key)
        if loaded is not None:
            return loaded
        if self.settings.preload_models:
            message = self.errors.get(artifact.key, "model_not_loaded")
            raise RuntimeError(message)
        return self._load(artifact)

    def keys(self) -> set[tuple[str, str]]:
        return set(self.loaded)

    def ready(self) -> bool:
        return bool(self.registry.artifacts) and len(self.loaded) == len(self.registry.artifacts) and not self.errors


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry, telemetry: Telemetry, outputs: OutputStore):
        self.settings = settings
        self.registry = registry
        self.telemetry = telemetry
        self.outputs = outputs
        self.cache = ModelCache(settings, registry)
        self.semaphore = __import__("threading").BoundedSemaphore(1)

    def startup(self) -> None:
        self.registry.load()
        if self.registry.errors:
            self.cache.startup = {"status": "registry_error", "error_count": len(self.registry.errors)}
            self.telemetry.startup = self.cache.startup
            return
        self.cache.preload()
        self.telemetry.startup = self.cache.startup

    def ready(self) -> bool:
        return self.cache.ready()

    def predict(self, request_id: str, artifact: ModelArtifact, image_bytes: bytes, content_type: str) -> dict[str, Any]:
        request_started = time.perf_counter_ns()
        queue_started = time.perf_counter_ns()
        acquired = self.semaphore.acquire(timeout=60)
        queue_wait = (time.perf_counter_ns() - queue_started) / 1_000_000
        if not acquired:
            raise TimeoutError("inference queue timeout")
        try:
            return self._predict_locked(request_id, artifact, image_bytes, content_type, request_started, queue_wait)
        finally:
            self.semaphore.release()

    def _predict_locked(self, request_id: str, artifact: ModelArtifact, image_bytes: bytes, content_type: str, request_started: int, queue_wait: float) -> dict[str, Any]:
        try:
            decode_started = time.perf_counter_ns()
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                width, height = decoded.size
                if max(width, height) > self.settings.max_image_dimension or width * height > self.settings.max_image_pixels:
                    raise ValueError("image dimensions exceed service limits")
                image = decoded.convert("RGB")
                image.load()
            decode_ms = elapsed_ms(decode_started)
            image_sha = __import__("hashlib").sha256(image_bytes).hexdigest()
            loaded = self.cache.get(artifact)
            prediction = loaded.model.predict(
                source=image,
                imgsz=artifact.image_size,
                batch=1,
                conf=artifact.conf,
                iou=artifact.nms_iou,
                max_det=artifact.max_det,
                device=self.settings.device,
                half=artifact.precision == "fp16",
                verbose=False,
                save=False,
            )[0]
            speed = getattr(prediction, "speed", {}) or {}
            detections = filter_detections(
                detections_from_result(prediction, artifact.labels),
                self.settings.display_conf,
            )
            render_started = time.perf_counter_ns()
            output_image = render_detections(image, detections)
            render_ms = elapsed_ms(render_started)
            encode_started = time.perf_counter_ns()
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format="PNG", optimize=False)
            encode_ms = elapsed_ms(encode_started)
            total_ms = elapsed_ms(request_started)
            resources = self.telemetry.resources()
            timing = {
                "queue_wait": round(queue_wait, 4),
                "decode": round(decode_ms, 4),
                "preprocess": round(float(speed.get("preprocess", 0.0)), 4),
                "inference": round(float(speed.get("inference", 0.0)), 4),
                "postprocess": round(float(speed.get("postprocess", 0.0)), 4),
                "render_output": round(render_ms, 4),
                "encode_output": round(encode_ms, 4),
                "total": round(total_ms, 4),
            }
            self.outputs.put(request_id, output_buffer.getvalue())
            result = {
                "request_id": request_id,
                "model": {
                    "id": artifact.model_id,
                    "sha256": artifact.artifact_sha256,
                    "backend": loaded.backend_identity,
                    "precision": artifact.precision,
                    "requested_precision": artifact.precision,
                    "dataset": artifact.dataset,
                    "variant": artifact.variant,
                    "seed": artifact.seed,
                    "image_size": artifact.image_size,
                },
                "image": {"width": image.width, "height": image.height, "content_type": content_type, "sha256": image_sha},
                "detections": detections,
                "timing_ms": timing,
                "timing_source": "ultralytics" if speed else "service_wall_clock",
                "throughput": {"fps_sequential": round(1000.0 / total_ms, 4) if total_ms else None},
                "output_image_url": f"/v1/requests/{request_id}/output.png",
                "protocol": {
                    "batch_size": 1,
                    "conf": artifact.conf,
                    "nms_iou": artifact.nms_iou,
                    "max_det": artifact.max_det,
                },
                "decision": {"display_conf": self.settings.display_conf},
            }
            self.telemetry.record(RequestRecord(request_id, utc_now(), artifact.model_id, artifact.precision, len(detections), total_ms, True))
            return result
        except Exception as exc:
            total_ms = elapsed_ms(request_started)
            self.telemetry.record(RequestRecord(request_id, utc_now(), artifact.model_id, artifact.precision, 0, total_ms, False, type(exc).__name__))
            raise


def assert_tensorrt_backend(model: Any, expected_provider: str) -> str:
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    format_name = str(getattr(backend, "format", "")).lower()
    backend_impl = getattr(backend, "backend", backend)
    implementation_name = type(backend_impl).__name__.lower()
    if format_name != "engine" or "tensorrt" not in implementation_name:
        raise RegistryError(
            f"expected Ultralytics TensorRT backend, got format={format_name!r}, implementation={implementation_name!r}"
        )
    actual_provider = "ultralytics_tensorrt_engine"
    if expected_provider != actual_provider:
        raise RegistryError(f"registered provider {expected_provider!r} does not match actual {actual_provider!r}")
    return actual_provider


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def filter_detections(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [detection for detection in detections if detection["confidence"] >= threshold]

def detections_from_result(result: Any, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    detections = []
    for box, score, category in zip(boxes, scores, classes):
        class_id = int(category)
        detections.append({
            "class_id": class_id,
            "class_name": labels[class_id] if class_id < len(labels) else str(class_id),
            "confidence": round(float(score), 6),
            "xyxy": [round(float(value), 3) for value in box],
        })
    return detections


def render_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        color = (40, 220, 120)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, round(min(image.size) / 400)))
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        text_y = max(0, y1 - 14)
        draw.text((x1, text_y), label, fill=color)
    return output
