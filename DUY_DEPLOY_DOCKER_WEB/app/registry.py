from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRECISIONS = ("fp16", "fp32")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    dataset: str
    model_family: str
    variant: str
    seed: int | None
    image_size: int
    artifact_path: Path
    artifact_sha256: str
    precision: str
    expected_provider: str
    source: dict[str, Any]
    labels: tuple[str, ...]
    conf: float
    nms_iou: float
    max_det: int
    validation: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return self.model_id, self.precision

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "dataset": self.dataset,
            "model_family": self.model_family,
            "variant": self.variant,
            "seed": self.seed,
            "image_size": self.image_size,
            "artifact_kind": "ultralytics_engine",
            "artifact_path": self.artifact_path.name,
            "sha256": self.artifact_sha256,
            "precision": self.precision,
            "expected_provider": self.expected_provider,
            "source": self.source,
            "labels": list(self.labels),
            "protocol": {"batch_size": 1, "conf": self.conf, "nms_iou": self.nms_iou, "max_det": self.max_det},
            "validation": self.validation,
        }


class ModelRegistry:
    def __init__(self, path: Path, model_root: Path):
        self.path = path
        self.model_root = model_root
        self.artifacts: dict[tuple[str, str], ModelArtifact] = {}
        self.errors: list[str] = []

    def load(self) -> None:
        self.artifacts.clear()
        self.errors.clear()
        if not self.path.is_file():
            self.errors.append(f"registry_not_found:{self.path}")
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.errors.append(f"registry_invalid:{exc}")
            return
        entries = payload.get("models", [])
        if not isinstance(entries, list):
            self.errors.append("registry_invalid:models must be a list")
            return
        for index, raw in enumerate(entries):
            try:
                artifact = self._parse(raw)
                if artifact.key in self.artifacts:
                    raise RegistryError(f"duplicate artifact {artifact.key}")
                self.artifacts[artifact.key] = artifact
            except (KeyError, TypeError, ValueError, RegistryError) as exc:
                self.errors.append(f"entry_{index}:{exc}")

    def _parse(self, raw: dict[str, Any]) -> ModelArtifact:
        if not isinstance(raw, dict):
            raise RegistryError("entry must be an object")
        precision = str(raw["precision"]).lower()
        if precision not in PRECISIONS:
            raise RegistryError(f"unsupported precision {precision}")
        artifact_text = str(raw["artifact_path"])
        artifact = Path(artifact_text)
        if artifact.is_absolute() or ".." in artifact.parts:
            raise RegistryError("artifact_path must be relative to the model root")
        model_root = self.model_root.resolve()
        artifact = (model_root / artifact).resolve(strict=False)
        try:
            artifact.relative_to(model_root)
        except ValueError as exc:
            raise RegistryError("artifact_path escapes the model root") from exc
        if artifact.suffix != ".engine":
            raise RegistryError("artifact_path must point to a .engine file")
        sha = str(raw["sha256"]).lower()
        if len(sha) != 64 or any(character not in "0123456789abcdef" for character in sha):
            raise RegistryError("sha256 must be a 64-character hexadecimal digest")
        source = dict(raw["source"])
        for field in ("hf_repo", "revision", "path"):
            if not str(source.get(field, "")).strip():
                raise RegistryError(f"source.{field} is required")
        validation = dict(raw["validation"])
        for field in ("status", "validated_at", "runtime"):
            if not str(validation.get(field, "")).strip():
                raise RegistryError(f"validation.{field} is required")
        labels = tuple(str(label) for label in raw.get("labels", ["ship"]))
        protocol = raw.get("protocol", {})
        if protocol.get("batch_size", 1) != 1:
            raise RegistryError("batch_size must be 1")
        if float(protocol.get("conf", 0.001)) != 0.001:
            raise RegistryError("conf must be 0.001")
        if protocol.get("nms_iou", 0.5) != 0.5:
            raise RegistryError("nms_iou must be 0.5")
        if int(protocol.get("max_det", 300)) != 300:
            raise RegistryError("max_det must be 300")
        return ModelArtifact(
            model_id=str(raw["id"]),
            dataset=str(raw["dataset"]),
            model_family=str(raw["model_family"]),
            variant=str(raw["variant"]),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            image_size=int(raw["image_size"]),
            artifact_path=artifact,
            artifact_sha256=sha,
            precision=precision,
            expected_provider=str(raw.get("expected_provider", "ultralytics_tensorrt")),
            source=dict(raw.get("source", {})),
            labels=labels,
            conf=float(protocol.get("conf", 0.001)),
            nms_iou=float(protocol.get("nms_iou", 0.5)),
            max_det=int(protocol.get("max_det", 300)),
            validation=dict(raw.get("validation", {})),
        )

    def get(self, model_id: str, precision: str) -> ModelArtifact | None:
        return self.artifacts.get((model_id, precision.lower()))

    def model_ids(self) -> list[str]:
        return sorted({artifact.model_id for artifact in self.artifacts.values()})

    def public_models(self, cache_keys: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        cache_keys = cache_keys or set()
        grouped: dict[str, dict[str, Any]] = {}
        for artifact in self.artifacts.values():
            row = grouped.setdefault(artifact.model_id, {
                "id": artifact.model_id,
                "dataset": artifact.dataset,
                "model_family": artifact.model_family,
                "variant": artifact.variant,
                "seed": artifact.seed,
                "image_size": artifact.image_size,
                "precisions": {},
            })
            item = artifact.public_dict()
            item["loaded"] = artifact.key in cache_keys
            item["available"] = artifact.artifact_path.is_file()
            row["precisions"][artifact.precision] = item
        return sorted(grouped.values(), key=lambda row: row["id"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(artifact: ModelArtifact) -> str:
    if not artifact.artifact_path.is_file():
        raise FileNotFoundError(artifact.artifact_path)
    actual = sha256(artifact.artifact_path)
    if actual != artifact.artifact_sha256:
        raise RegistryError(f"sha256 mismatch for {artifact.artifact_path}: expected {artifact.artifact_sha256}, got {actual}")
    return actual
