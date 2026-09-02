"""Registry for project-specific Ultralytics modules.

The registry is deliberately separate from Ultralytics' parser. A small adapter
can inject this mapping when parsing a project model YAML, while the upstream
package remains untouched.
"""

from __future__ import annotations

from collections.abc import MutableMapping

from .losses import BoundaryContrastiveLoss, LocalizationQualityLoss, WiseIouLoss
from .modules import (
    AmplitudePerturbation,
    AugmentationAwareEvidence,
    BiLevelRoutingAttention,
    CBAM,
    C2fCBAM,
    C3CBAM,
    ChannelAttention,
    C2f_PConv,
    C2fNAT,
    DedicatedCueSlots,
    DetachedResidualFusion,
    FasterNetBlock,
    FullSelfAttention,
    GlobalChannelContextCalibration,
    GTChannelSpecialization,
    GTCuePreservationHead,
    GradientIsolatedEvidence,
    InputCueBank,
    InputCueConv,
    KVCompressedAttention,
    KVCompressedAttentionPartial,
    KVCompressedTransformerEncoder,
    LearnableGlobalScalar,
    LocalContrastBasisStem,
    MultiCueEvidenceFusion,
    MatchedChannelPerturbation,
    P2AmplitudeCalibrator,
    P2FeatureProbe,
    NATBlock,
    PatchKVCompressedAttention,
    PConv,
    RawColorSlotFusion,
    RawImageCueBank,
    ReceptanceKVCompressedAttention,
    ResidualDWConv,
    ResidualDWConv5,
    ScaleDisappearanceEvidence,
    SidecarResidualFusionStem,
    SingleContrastFormationStem,
    SplitChannelDetect,
    SpatialAttention,
    SurgicalPartialKVCompressedAttention,
    TopKAdaptiveGroupKVAttention,
    TopKGlobalGroupKVAttention,
)

CUSTOM_MODULES = {
    cls.__name__: cls
    for cls in (
        AugmentationAwareEvidence,
        AmplitudePerturbation,
        BiLevelRoutingAttention,
        CBAM,
        C2fCBAM,
        C3CBAM,
        ChannelAttention,
        C2f_PConv,
        C2fNAT,
        DedicatedCueSlots,
        DetachedResidualFusion,
        FasterNetBlock,
        FullSelfAttention,
        GlobalChannelContextCalibration,
        GTChannelSpecialization,
        GTCuePreservationHead,
        GradientIsolatedEvidence,
        InputCueBank,
        InputCueConv,
        KVCompressedAttention,
        KVCompressedAttentionPartial,
        KVCompressedTransformerEncoder,
        LearnableGlobalScalar,
        LocalContrastBasisStem,
        MultiCueEvidenceFusion,
        MatchedChannelPerturbation,
        P2AmplitudeCalibrator,
        P2FeatureProbe,
        NATBlock,
        PatchKVCompressedAttention,
        PConv,
        RawColorSlotFusion,
        RawImageCueBank,
        ReceptanceKVCompressedAttention,
        ResidualDWConv,
        ResidualDWConv5,
        ScaleDisappearanceEvidence,
        SidecarResidualFusionStem,
        SingleContrastFormationStem,
        SplitChannelDetect,
        SpatialAttention,
        SurgicalPartialKVCompressedAttention,
        TopKAdaptiveGroupKVAttention,
        TopKGlobalGroupKVAttention,
    )
}

CUSTOM_LOSSES = {
    cls.__name__: cls for cls in (BoundaryContrastiveLoss, LocalizationQualityLoss, WiseIouLoss)
}


def get_custom_module(name: str):
    """Return a project module by YAML class name or raise a useful error."""
    try:
        return CUSTOM_MODULES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CUSTOM_MODULES))
        raise KeyError(f"Unknown project module {name!r}. Known modules: {known}") from exc


def get_custom_loss(name: str):
    """Return a project loss by name for a future trainer/loss adapter."""
    try:
        return CUSTOM_LOSSES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CUSTOM_LOSSES))
        raise KeyError(f"Unknown project loss {name!r}. Known losses: {known}") from exc


def install_custom_modules(namespace: MutableMapping[str, object]) -> None:
    """Install project modules into a parser namespace without touching upstream."""
    namespace.update(CUSTOM_MODULES)
