"""Project-specific Ultralytics extension boundary.

Custom modules should move here incrementally. The legacy fork remains available
until all experiment entrypoints use this boundary.
"""

from .parser import load_project_model

__all__ = ("load_project_model",)
