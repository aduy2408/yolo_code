"""Registry for project-specific Ultralytics modules.

The registry is deliberately separate from Ultralytics' parser. A small adapter
can inject this mapping when parsing a project model YAML, while the upstream
package remains untouched.
"""

from __future__ import annotations

from collections.abc import MutableMapping

from .modules import (
    AugmentationAwareEvidence,
    C2f_PConv,
    DedicatedCueSlots,
    DetachedResidualFusion,
    FasterNetBlock,
    GTChannelSpecialization,
    GTCuePreservationHead,
    GradientIsolatedEvidence,
    InputCueBank,
    InputCueConv,
    LocalContrastBasisStem,
    MultiCueEvidenceFusion,
    PConv,
    RawColorSlotFusion,
    RawImageCueBank,
    ScaleDisappearanceEvidence,
    SidecarResidualFusionStem,
    SingleContrastFormationStem,
    SplitChannelDetect,
)

CUSTOM_MODULES = {
    cls.__name__: cls
    for cls in (
        AugmentationAwareEvidence,
        C2f_PConv,
        DedicatedCueSlots,
        DetachedResidualFusion,
        FasterNetBlock,
        GTChannelSpecialization,
        GTCuePreservationHead,
        GradientIsolatedEvidence,
        InputCueBank,
        InputCueConv,
        LocalContrastBasisStem,
        MultiCueEvidenceFusion,
        PConv,
        RawColorSlotFusion,
        RawImageCueBank,
        ScaleDisappearanceEvidence,
        SidecarResidualFusionStem,
        SingleContrastFormationStem,
        SplitChannelDetect,
    )
}


def get_custom_module(name: str):
    """Return a project module by YAML class name or raise a useful error."""
    try:
        return CUSTOM_MODULES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CUSTOM_MODULES))
        raise KeyError(f"Unknown project module {name!r}. Known modules: {known}") from exc


def install_custom_modules(namespace: MutableMapping[str, object]) -> None:
    """Install project modules into a parser namespace without touching upstream."""
    namespace.update(CUSTOM_MODULES)
