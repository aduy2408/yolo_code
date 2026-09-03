"""Canonical, dependency-free architecture descriptions for YOLOv8-v11.

These specs mirror the checked-in Ultralytics YAMLs and are intentionally
separate from the runtime parser. They are useful for documentation, graph
inspection, and regression tests without importing the legacy fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayerSpec:
    """One YAML-style layer: ``[from, repeats, module, args]``."""

    source: int | tuple[int, ...]
    repeats: int
    module: str
    args: tuple[Any, ...] = ()
    stage: str | None = None


@dataclass(frozen=True)
class ArchitectureSpec:
    """A compact detection architecture description."""

    name: str
    outputs: tuple[str, ...]
    backbone: tuple[LayerSpec, ...]
    head: tuple[LayerSpec, ...]
    notes: tuple[str, ...] = ()

    @property
    def layers(self) -> tuple[LayerSpec, ...]:
        return self.backbone + self.head


# The following graphs are the compact n/t configurations used by the
# repository's canonical YAMLs. Channel counts are pre compound-scaling.
YOLOV8 = ArchitectureSpec(
    name="YOLOv8",
    outputs=("P3/8", "P4/16", "P5/32"),
    backbone=(
        LayerSpec(-1, 1, "Conv", (64, 3, 2), "P1/2"),
        LayerSpec(-1, 1, "Conv", (128, 3, 2), "P2/4"),
        LayerSpec(-1, 3, "C2f", (128, True)),
        LayerSpec(-1, 1, "Conv", (256, 3, 2), "P3/8"),
        LayerSpec(-1, 6, "C2f", (256, True)),
        LayerSpec(-1, 1, "Conv", (512, 3, 2), "P4/16"),
        LayerSpec(-1, 6, "C2f", (512, True)),
        LayerSpec(-1, 1, "Conv", (1024, 3, 2), "P5/32"),
        LayerSpec(-1, 3, "C2f", (1024, True)),
        LayerSpec(-1, 1, "SPPF", (1024, 5)),
    ),
    head=(
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 6), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (512,)),
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 4), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (256,), "P3/8"),
        LayerSpec(-1, 1, "Conv", (256, 3, 2)),
        LayerSpec((-1, 12), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (512,), "P4/16"),
        LayerSpec(-1, 1, "Conv", (512, 3, 2)),
        LayerSpec((-1, 9), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (1024,), "P5/32"),
        LayerSpec((15, 18, 21), 1, "Detect", ("nc",)),
    ),
    notes=("Decoupled anchor-free Detect head.", "C2f backbone and PAN-FPN neck."),
)

YOLOV9 = ArchitectureSpec(
    name="YOLOv9",
    outputs=("P3/8", "P4/16", "P5/32"),
    backbone=(
        LayerSpec(-1, 1, "Conv", (16, 3, 2), "P1/2"),
        LayerSpec(-1, 1, "Conv", (32, 3, 2), "P2/4"),
        LayerSpec(-1, 1, "ELAN1", (32, 32, 16)),
        LayerSpec(-1, 1, "AConv", (64,), "P3/8"),
        LayerSpec(-1, 1, "RepNCSPELAN4", (64, 64, 32, 3)),
        LayerSpec(-1, 1, "AConv", (96,), "P4/16"),
        LayerSpec(-1, 1, "RepNCSPELAN4", (96, 96, 48, 3)),
        LayerSpec(-1, 1, "AConv", (128,), "P5/32"),
        LayerSpec(-1, 1, "RepNCSPELAN4", (128, 128, 64, 3)),
        LayerSpec(-1, 1, "SPPELAN", (128, 64)),
    ),
    head=(
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 6), 1, "Concat", (1,)),
        LayerSpec(-1, 1, "RepNCSPELAN4", (96, 96, 48, 3)),
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 4), 1, "Concat", (1,)),
        LayerSpec(-1, 1, "RepNCSPELAN4", (64, 64, 32, 3), "P3/8"),
        LayerSpec(-1, 1, "AConv", (48,)),
        LayerSpec((-1, 12), 1, "Concat", (1,)),
        LayerSpec(-1, 1, "RepNCSPELAN4", (96, 96, 48, 3), "P4/16"),
        LayerSpec(-1, 1, "AConv", (64,)),
        LayerSpec((-1, 9), 1, "Concat", (1,)),
        LayerSpec(-1, 1, "RepNCSPELAN4", (128, 128, 64, 3), "P5/32"),
        LayerSpec((15, 18, 21), 1, "Detect", ("nc",)),
    ),
    notes=(
        "GELAN backbone with RepNCSPELAN4 and SPPELAN.",
        "AConv performs average-pool plus stride-2 convolution.",
        "PGI is a training-time gradient/progressive-information mechanism, not an extra inference feature-map stage.",
    ),
)

YOLOV10 = ArchitectureSpec(
    name="YOLOv10",
    outputs=("P3/8", "P4/16", "P5/32"),
    backbone=(
        LayerSpec(-1, 1, "Conv", (64, 3, 2), "P1/2"),
        LayerSpec(-1, 1, "Conv", (128, 3, 2), "P2/4"),
        LayerSpec(-1, 3, "C2f", (128, True)),
        LayerSpec(-1, 1, "Conv", (256, 3, 2), "P3/8"),
        LayerSpec(-1, 6, "C2f", (256, True)),
        LayerSpec(-1, 1, "SCDown", (512, 3, 2), "P4/16"),
        LayerSpec(-1, 6, "C2f", (512, True)),
        LayerSpec(-1, 1, "SCDown", (1024, 3, 2), "P5/32"),
        LayerSpec(-1, 3, "C2f", (1024, True)),
        LayerSpec(-1, 1, "SPPF", (1024, 5)),
        LayerSpec(-1, 1, "PSA", (1024,)),
    ),
    head=(
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 6), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (512,)),
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 4), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (256,), "P3/8"),
        LayerSpec(-1, 1, "Conv", (256, 3, 2)),
        LayerSpec((-1, 13), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2f", (512,), "P4/16"),
        LayerSpec(-1, 1, "SCDown", (512, 3, 2)),
        LayerSpec((-1, 10), 1, "Concat", (1,)),
        LayerSpec(-1, 3, "C2fCIB", (1024, True, True), "P5/32"),
        LayerSpec((16, 19, 22), 1, "v10Detect", ("nc",)),
    ),
    notes=(
        "SCDown replaces selected stride-2 Conv layers with spatial/channel decoupled downsampling.",
        "PSA adds partial self-attention at the deepest backbone stage.",
        "v10Detect has one-to-many and one-to-one branches for dual-assignment training and NMS-free inference.",
    ),
)

YOLOV11 = ArchitectureSpec(
    name="YOLO11",
    outputs=("P3/8", "P4/16", "P5/32"),
    backbone=(
        LayerSpec(-1, 1, "Conv", (64, 3, 2), "P1/2"),
        LayerSpec(-1, 1, "Conv", (128, 3, 2), "P2/4"),
        LayerSpec(-1, 2, "C3k2", (256, False, 0.25)),
        LayerSpec(-1, 1, "Conv", (256, 3, 2), "P3/8"),
        LayerSpec(-1, 2, "C3k2", (512, False, 0.25)),
        LayerSpec(-1, 1, "Conv", (512, 3, 2), "P4/16"),
        LayerSpec(-1, 2, "C3k2", (512, True)),
        LayerSpec(-1, 1, "Conv", (1024, 3, 2), "P5/32"),
        LayerSpec(-1, 2, "C3k2", (1024, True)),
        LayerSpec(-1, 1, "SPPF", (1024, 5)),
        LayerSpec(-1, 2, "C2PSA", (1024,)),
    ),
    head=(
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 6), 1, "Concat", (1,)),
        LayerSpec(-1, 2, "C3k2", (512, False)),
        LayerSpec(-1, 1, "Upsample", (None, 2, "nearest")),
        LayerSpec((-1, 4), 1, "Concat", (1,)),
        LayerSpec(-1, 2, "C3k2", (256, False), "P3/8"),
        LayerSpec(-1, 1, "Conv", (256, 3, 2)),
        LayerSpec((-1, 13), 1, "Concat", (1,)),
        LayerSpec(-1, 2, "C3k2", (512, False), "P4/16"),
        LayerSpec(-1, 1, "Conv", (512, 3, 2)),
        LayerSpec((-1, 10), 1, "Concat", (1,)),
        LayerSpec(-1, 2, "C3k2", (1024, True), "P5/32"),
        LayerSpec((16, 19, 22), 1, "Detect", ("nc",)),
    ),
    notes=(
        "C3k2 is a lighter CSP-style block with optional C3k inner bottlenecks.",
        "C2PSA applies partial spatial attention at the deepest stage.",
        "The detection head remains the standard anchor-free Detect family.",
    ),
)

ARCHITECTURES = {spec.name.lower(): spec for spec in (YOLOV8, YOLOV9, YOLOV10, YOLOV11)}
ARCHITECTURES.update({"yolov8": YOLOV8, "yolov9": YOLOV9, "yolov10": YOLOV10, "yolo11": YOLOV11})


def get_architecture(name: str) -> ArchitectureSpec:
    """Return a canonical architecture by case-insensitive name."""

    key = name.lower().replace("-", "")
    aliases = {"yolo8": "yolov8", "yolo9": "yolov9", "yolo10": "yolov10", "yolo11": "yolo11"}
    try:
        return ARCHITECTURES[aliases.get(key, key)]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture {name!r}; choose YOLOv8, YOLOv9, YOLOv10, or YOLO11") from exc


def graph_lines(spec: ArchitectureSpec) -> list[str]:
    """Render a readable layer listing suitable for reports and debugging."""

    lines = [f"{spec.name}: outputs={', '.join(spec.outputs)}", "backbone:"]
    for index, layer in enumerate(spec.backbone):
        lines.append(f"  B{index:02d}: {layer.module} x{layer.repeats} args={list(layer.args)} {layer.stage or ''}".rstrip())
    lines.append("head:")
    offset = len(spec.backbone)
    for index, layer in enumerate(spec.head, offset):
        lines.append(f"  H{index:02d}: {layer.module} x{layer.repeats} args={list(layer.args)} {layer.stage or ''}".rstrip())
    return lines


__all__ = [
    "ARCHITECTURES",
    "ArchitectureSpec",
    "LayerSpec",
    "YOLOV8",
    "YOLOV9",
    "YOLOV10",
    "YOLOV11",
    "get_architecture",
    "graph_lines",
]
