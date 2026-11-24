"""Project-specific Ultralytics extension boundary.

Custom modules should move here incrementally. The legacy fork remains available
until all experiment entrypoints use this boundary.
"""

__all__ = ("load_project_model",)


def __getattr__(name: str):
    """Load the parser bridge lazily to keep loss-only imports acyclic."""
    if name == "load_project_model":
        from .parser import load_project_model

        return load_project_model
    raise AttributeError(name)
