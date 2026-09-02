"""Version-matched parser bridge for project-owned YAML modules."""

from __future__ import annotations

import contextlib
import inspect
import textwrap
from types import ModuleType
from typing import Any

from .modules import cue_channels
from .registry import CUSTOM_MODULES


_IMAGE_AWARE_MODULES = frozenset(
    {
        "GradientIsolatedEvidence",
        "AugmentationAwareEvidence",
        "RawImageCueBank",
        "RawColorSlotFusion",
        "MultiCueEvidenceFusion",
        "DedicatedCueSlots",
        "DetachedResidualFusion",
    }
)


def _project_parse_model(upstream_tasks: ModuleType):
    """Build an upstream-compatible parser with project layer rules.

    The upstream function is copied at runtime from the pinned package and
    extended in this project namespace. This keeps the vendor submodule
    untouched while avoiding a second permanently forked parser implementation.
    """
    source = textwrap.dedent(inspect.getsource(upstream_tasks.parse_model))
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
        elif m is InputCueConv:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is InputCueBank:
            c2 = cue_channels(args[0])
        elif m in {GradientIsolatedEvidence, AugmentationAwareEvidence}:
            c1 = ch[f]
            evidence_ch = int(args[0]) if args else 8
            c2 = c1 + evidence_ch
            args = [c1, *args]
        elif m in {RawImageCueBank, RawColorSlotFusion, MultiCueEvidenceFusion, DedicatedCueSlots, DetachedResidualFusion}:
            c2 = 4 if m is RawImageCueBank else 32
        elif m is ScaleDisappearanceEvidence:
            c1 = [ch[x] for x in f]
            c2 = int(args[0]) if args else 8
            args = [*c1, *args]
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

    with project_parser(tasks), project_runtime():
        result = YOLO(model, task=task, verbose=verbose)
    _install_instance_runtime(result.model)
    return result


def _project_predict_once(upstream_base_model: type):
    """Build a runtime loop that passes the original image to image-aware layers."""
    source = textwrap.dedent(inspect.getsource(upstream_base_model._predict_once))
    source = source.replace("def _predict_once(", "def _project_predict_once(", 1)
    source = source.replace("    y, dt, embeddings = [], [], []  # outputs\n", "    img0 = x\n    y, dt, embeddings = [], [], []  # outputs\n", 1)
    old = "        x = m(x)  # run\n"
    new = (
        "        x = m(x, img0) if m.__class__.__name__ in _IMAGE_AWARE_MODULES else m(x)  # run\n"
    )
    if old not in source:
        raise RuntimeError("Unsupported upstream _predict_once layout")
    source = source.replace(old, new, 1)
    namespace = dict(vars(upstream_base_model.__module__ and __import__(upstream_base_model.__module__, fromlist=["*"])))
    namespace["_IMAGE_AWARE_MODULES"] = _IMAGE_AWARE_MODULES
    exec(compile(source, "<project _predict_once>", "exec"), namespace)  # nosec B102: pinned upstream source
    return namespace["_project_predict_once"]


def _install_instance_runtime(model: Any) -> None:
    """Install image-aware prediction only on this model instance."""
    from types import MethodType
    from ultralytics.nn.tasks import BaseModel

    model._project_original_predict_once = model._predict_once
    model._predict_once = MethodType(_project_predict_once(BaseModel), model)


@contextlib.contextmanager
def project_runtime():
    """Temporarily enable image-aware layers for newly built upstream models."""
    from ultralytics.nn.tasks import BaseModel

    original = BaseModel._predict_once
    BaseModel._predict_once = _project_predict_once(BaseModel)
    try:
        yield
    finally:
        BaseModel._predict_once = original
