"""Project-specific neural modules built on clean Ultralytics primitives."""

from .evidence import AugmentationAwareEvidence, GradientIsolatedEvidence, ScaleDisappearanceEvidence
from .gt_cue_loss import (
    DedicatedCueSlots,
    DetachedResidualFusion,
    GTChannelSpecialization,
    GTCuePreservationHead,
    SplitChannelDetect,
)
from .input_cue import InputCueBank
from .local_contrast import LocalContrastBasisStem, SingleContrastFormationStem
from .pconv import C2f_PConv, FasterNetBlock, PConv
from .raw_cue_fusion import MultiCueEvidenceFusion, RawColorSlotFusion, RawImageCueBank

__all__ = (
    "AugmentationAwareEvidence",
    "C2f_PConv",
    "DedicatedCueSlots",
    "DetachedResidualFusion",
    "FasterNetBlock",
    "GTChannelSpecialization",
    "GTCuePreservationHead",
    "GradientIsolatedEvidence",
    "InputCueBank",
    "LocalContrastBasisStem",
    "MultiCueEvidenceFusion",
    "PConv",
    "RawColorSlotFusion",
    "RawImageCueBank",
    "ScaleDisappearanceEvidence",
    "SingleContrastFormationStem",
    "SplitChannelDetect",
)
