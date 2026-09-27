"""Shared data, hook, and artifact utilities for TOD probes.

The code in this package is deliberately side-effect free with respect to the
detector. It loads a checkpoint, registers temporary hooks, and writes
artifacts under a caller-provided output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    image_glob: str
    label_mode: str = "yolo"
    annotation_dir: str | None = None
    split: str = "test"


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 42
    imgsz: int = 640
    max_images: int = 32
    device: str = "cpu"
    conf: float = 0.001
    feature_layers: tuple[str, ...] = ("model.4", "model.8", "model.12", "model.15", "model.18", "model.21")


DATASETS = {
    # The checked-in YOLO split contains broken absolute symlinks in this
    # checkout. Use the canonical local source tree for diagnostics and record
    # that this is the all-source protocol rather than silently calling it test.
    "levir-ship": DatasetSpec(
        "LEVIR-Ship", ROOT / "datasets/LevirShipData", "All Images/*", annotation_dir="All Annotations", split="all_source"
    ),
    "visdrone": DatasetSpec(
        "VisDrone2019-DET",
        ROOT / "datasets/VisDrone2019/VisDrone2019-DET-train",
        "images/*",
        label_mode="visdrone",
        annotation_dir="annotations",
        split="train",
    ),
    "tinyperson": DatasetSpec(
        "TinyPerson", ROOT / "datasets/tinyperson_split_42_corner_sw640_sh512", "images/test/*"
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_paths(spec: DatasetSpec, limit: int) -> list[Path]:
    paths = sorted(path for path in spec.root.glob(spec.image_glob) if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images found for {spec.name}: {spec.root / spec.image_glob}")
    annotated = [path for path in paths if label_path(path, spec).is_file() and label_path(path, spec).stat().st_size > 0]
    return (annotated or paths)[:limit]


def label_path(image: Path, spec: DatasetSpec) -> Path:
    if spec.annotation_dir and spec.label_mode == "yolo" and "All Images" in image.parts:
        return spec.root / spec.annotation_dir / f"{image.stem}.txt"
    if spec.label_mode == "visdrone":
        return spec.root / (spec.annotation_dir or "annotations") / f"{image.stem}.txt"
    parts = list(image.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_boxes(image: Path, spec: DatasetSpec) -> list[dict[str, float]]:
    frame = cv2.imread(str(image))
    if frame is None:
        raise ValueError(f"Could not read image: {image}")
    height, width = frame.shape[:2]
    path = label_path(image, spec)
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [item.strip() for item in line.replace(",", " ").split() if item.strip()]
        if spec.label_mode == "visdrone":
            if len(values) < 6:
                continue
            x, y, bw, bh = map(float, values[:4])
            category = int(float(values[5]))
            if category == 0 or bw <= 0 or bh <= 0:
                continue
            rows.append({"x1": x, "y1": y, "x2": x + bw, "y2": y + bh, "class": category - 1})
        elif len(values) >= 8 and "," in line:
            x, y, bw, bh = map(float, values[:4])
            if bw > 0 and bh > 0:
                rows.append({"x1": x, "y1": y, "x2": x + bw, "y2": y + bh, "class": 0})
        elif len(values) >= 5:
            category, cx, cy, bw, bh = map(float, values[:5])
            bw, bh = bw * width, bh * height
            cx, cy = cx * width, cy * height
            rows.append({"x1": cx - bw / 2, "y1": cy - bh / 2, "x2": cx + bw / 2, "y2": cy + bh / 2, "class": category})
    return rows


def size_bucket(box: dict[str, float]) -> str:
    diagonal = float(np.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"]))
    if diagonal < 8:
        return "tiny_lt8"
    if diagonal < 16:
        return "tiny_8_16"
    if diagonal < 32:
        return "small_16_32"
    return "medium_ge32"


def load_yolo(checkpoint: Path, device: str) -> Any:
    # Probe code must use the clean pinned upstream package. The historical
    # compatibility fork has project-specific Detect changes and cannot load
    # the report-listed baseline checkpoint reliably.
    package = ROOT / "vendor/ultralytics_upstream"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    model.model.to(device).eval()
    return model


def module_map(model: Any) -> dict[str, Any]:
    return dict(model.model.named_modules())


def tensor_from_output(output: Any) -> Any:
    import torch

    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item):
                return item
    return None


def feature_summary(feature: Any, box: dict[str, float], image_shape: tuple[int, int]) -> dict[str, float]:
    import torch

    tensor = tensor_from_output(feature)
    if tensor is None:
        return {"energy": float("nan"), "object_energy": float("nan"), "context_energy": float("nan"), "object_fraction": float("nan")}
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        flat = tensor.detach().float().reshape(-1)
        energy = float(flat.abs().mean().cpu())
        return {"energy": energy, "object_energy": energy, "context_energy": energy, "object_fraction": 1.0}
    height, width = image_shape
    mask = torch.zeros((tensor.shape[-2], tensor.shape[-1]), dtype=torch.bool, device=tensor.device)
    x1 = max(0, min(mask.shape[1] - 1, round(box["x1"] / max(width, 1) * mask.shape[1])))
    x2 = max(x1 + 1, min(mask.shape[1], round(box["x2"] / max(width, 1) * mask.shape[1])))
    y1 = max(0, min(mask.shape[0] - 1, round(box["y1"] / max(height, 1) * mask.shape[0])))
    y2 = max(y1 + 1, min(mask.shape[0], round(box["y2"] / max(height, 1) * mask.shape[0])))
    mask[y1:y2, x1:x2] = True
    values = tensor.detach().float().abs()[0]
    energy = float(values.mean().cpu())
    object_energy = float(values[:, mask].mean().cpu())
    context_energy = float(values[:, ~mask].mean().cpu()) if (~mask).any() else 0.0
    return {"energy": energy, "object_energy": object_energy, "context_energy": context_energy, "object_fraction": object_energy / max(energy, 1e-8)}


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def metadata(spec: DatasetSpec, config: ProbeConfig, checkpoint: Path, probe: str) -> dict[str, Any]:
    return {
        "probe": probe,
        "dataset": asdict(spec) | {"root": str(spec.root)},
        "config": asdict(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint) if checkpoint.is_file() else "missing",
        "source_commit": git_commit(),
        "detector_behavior_modified": False,
    }
