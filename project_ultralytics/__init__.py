"""Project-specific Ultralytics extension boundary.

Custom modules should move here incrementally. The legacy fork remains available
until all experiment entrypoints use this boundary.
"""

from .architectures import (
    ARCHITECTURES,
    YOLOV8,
    YOLOV9,
    YOLOV10,
    YOLOV11,
    get_architecture,
)


def load_project_model(*args, **kwargs):
    """Load a runtime model while keeping metadata imports dependency-free."""

    from .parser import load_project_model as _load_project_model

    return _load_project_model(*args, **kwargs)

__all__ = (
    "ARCHITECTURES",
    "YOLOV8",
    "YOLOV9",
    "YOLOV10",
    "YOLOV11",
    "get_architecture",
    "load_project_model",
)
