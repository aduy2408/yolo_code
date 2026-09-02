"""Version-matched parser bridge for project-owned YAML modules."""

from __future__ import annotations

import contextlib
import inspect
from types import ModuleType
from typing import Any

from .registry import CUSTOM_MODULES


def _project_parse_model(upstream_tasks: ModuleType):
    """Build an upstream-compatible parser with project layer rules.

    The upstream function is copied at runtime from the pinned package and
    extended in this project namespace. This keeps the vendor submodule
    untouched while avoiding a second permanently forked parser implementation.
    """
    source = inspect.getsource(upstream_tasks.parse_model)
    source = source.replace("def parse_model(", "def parse_model_project(", 1)
    marker = "        elif m is AIFI:\n"
    branch = """        elif m is WeightedAdd:
            if not isinstance(f, list) or not f:
                raise ValueError(\"WeightedAdd YAML layer requires a non-empty list of input indices\")
            c2 = ch[f[0]]
            args = [len(f)]
        elif m is ChannelAttention:
            c2 = ch[f]
            args = [c2, *args]
        elif m is CBAM:
            c2 = ch[f]
            args = [c2, *args]
        elif m is SpatialAttention:
            c2 = ch[f]
        elif m is KVCompressedAttention:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {C2fCBAM, C3CBAM, C2f_PConv, C2fNAT}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)
            n = 1
        elif m is NATBlock:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {
            BiLevelRoutingAttention,
            FullSelfAttention,
            PatchKVCompressedAttention,
            ReceptanceKVCompressedAttention,
            KVCompressedAttentionPartial,
            SurgicalPartialKVCompressedAttention,
            KVCompressedTransformerEncoder,
            TopKAdaptiveGroupKVAttention,
            TopKGlobalGroupKVAttention,
        }:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in {P2FeatureProbe, P2AmplitudeCalibrator, MatchedChannelPerturbation, ResidualDWConv, ResidualDWConv5}:
            c2 = ch[f]
            args = [c2, *args]
"""
    if marker not in source:
        raise RuntimeError("Unsupported upstream parse_model layout: WeightedAdd insertion point not found")
    source = source.replace(marker, branch + marker, 1)
    namespace = dict(vars(upstream_tasks))
    namespace.update(CUSTOM_MODULES)
    exec(compile(source, "<project parse_model>", "exec"), namespace)  # nosec B102: pinned upstream source
    return namespace["parse_model_project"]


@contextlib.contextmanager
def project_parser(upstream_tasks: ModuleType):
    """Temporarily install the project parser while a model is constructed."""
    parser = _project_parse_model(upstream_tasks)
    original = upstream_tasks.parse_model
    upstream_tasks.parse_model = parser
    try:
        yield
    finally:
        upstream_tasks.parse_model = original


def load_project_model(model: Any, *, task: str | None = None, verbose: bool = True):
    """Load a YAML/checkpoint through clean upstream plus project modules."""
    from ultralytics import YOLO
    from ultralytics.nn import tasks

    with project_parser(tasks):
        return YOLO(model, task=task, verbose=verbose)
