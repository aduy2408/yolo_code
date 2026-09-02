"""Project-specific neural modules built on clean Ultralytics primitives."""

from .evidence import AugmentationAwareEvidence, GradientIsolatedEvidence, ScaleDisappearanceEvidence
from .feature_calibration import (
    AmplitudePerturbation,
    LearnableGlobalScalar,
    MatchedChannelPerturbation,
    P2AmplitudeCalibrator,
    P2FeatureProbe,
    ResidualDWConv,
    ResidualDWConv5,
)
from .gt_cue_loss import (
    DedicatedCueSlots,
    DetachedResidualFusion,
    GTChannelSpecialization,
    GTCuePreservationHead,
    SplitChannelDetect,
)
from .input_cue import InputCueBank, InputCueConv
from .local_contrast import LocalContrastBasisStem, SidecarResidualFusionStem, SingleContrastFormationStem
from .pconv import C2f_PConv, FasterNetBlock, PConv
from .raw_cue_fusion import MultiCueEvidenceFusion, RawColorSlotFusion, RawImageCueBank

__all__ = (
    "AugmentationAwareEvidence",
    "AmplitudePerturbation",
    "C2f_PConv",
    "DedicatedCueSlots",
    "DetachedResidualFusion",
    "FasterNetBlock",
    "GTChannelSpecialization",
    "GTCuePreservationHead",
    "GradientIsolatedEvidence",
    "InputCueBank",
    "InputCueConv",
    "LearnableGlobalScalar",
    "LocalContrastBasisStem",
    "MultiCueEvidenceFusion",
    "MatchedChannelPerturbation",
    "P2AmplitudeCalibrator",
    "P2FeatureProbe",
    "PConv",
    "RawColorSlotFusion",
    "RawImageCueBank",
    "ResidualDWConv",
    "ResidualDWConv5",
    "ScaleDisappearanceEvidence",
    "SidecarResidualFusionStem",
    "SingleContrastFormationStem",
    "SplitChannelDetect",
)
