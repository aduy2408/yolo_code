from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from app.inference import filter_detections
from app.output_store import OutputStore
from app.registry import ModelRegistry, RegistryError, sha256, verify_artifact
from app.telemetry import percentile


class RegistryTests(unittest.TestCase):
    def test_registry_loads_engine_by_precision_and_verifies_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "model.engine"
            engine.write_bytes(b"engine")
            payload = {
                "models": [{
                    "id": "demo",
                    "dataset": "levirship",
                    "model_family": "yolov8",
                    "variant": "oacp-r2",
                    "seed": 43,
                    "image_size": 512,
                    "artifact_path": "model.engine",
                    "sha256": sha256(engine),
                    "precision": "fp16",
                    "expected_provider": "ultralytics_tensorrt_engine",
                    "labels": ["ship"],
                    "protocol": {"batch_size": 1, "conf": 0.001, "nms_iou": 0.5, "max_det": 300},
                    "source": {"hf_repo": "repo", "revision": "rev", "path": "weights/best.pt"},
                    "validation": {"status": "validated_on_target", "validated_at": "2026-11-07T00:00:00Z", "runtime": "test"},
                }]
            }
            registry_path = root / "models.json"
            registry_path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ModelRegistry(registry_path, root)
            registry.load()
            self.assertFalse(registry.errors)
            artifact = registry.get("demo", "fp16")
            self.assertIsNotNone(artifact)
            self.assertEqual(verify_artifact(artifact), hashlib.sha256(b"engine").hexdigest())

    def test_registry_rejects_non_protocol_iou(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "models.json"
            payload = {
                "models": [{
                    "id": "bad",
                    "dataset": "levirship",
                    "model_family": "yolov8",
                    "variant": "oacp-r2",
                    "seed": 43,
                    "image_size": 512,
                    "artifact_path": "x.engine",
                    "sha256": "0" * 64,
                    "precision": "fp16",
                    "source": {"hf_repo": "repo", "revision": "rev", "path": "weights/best.pt"},
                    "validation": {"status": "validated_on_target", "validated_at": "2026-11-07T00:00:00Z", "runtime": "test"},
                    "protocol": {"batch_size": 1, "conf": 0.001, "nms_iou": 0.4, "max_det": 300},
                }]
            }
            registry_path.write_text(json.dumps(payload), encoding="utf-8")
            registry = ModelRegistry(registry_path, root)
            registry.load()
            self.assertTrue(any("nms_iou" in error for error in registry.errors))


class UtilityTests(unittest.TestCase):
    def test_display_confidence_filters_candidates(self) -> None:
        detections = [{"confidence": 0.29}, {"confidence": 0.3}, {"confidence": 0.91}]
        self.assertEqual(filter_detections(detections, 0.3), detections[1:])

    def test_percentile_is_bounded(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.0)
        self.assertIsNone(percentile([], 0.95))
    def test_output_store_ttl_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OutputStore(Path(directory), 60)
            request_id = "123e4567-e89b-12d3-a456-426614174000"
            path = store.put(request_id, b"png")
            self.assertEqual(store.get(request_id), path)
            self.assertIsNone(store.get("../secret"))
            self.assertEqual(path.read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()
