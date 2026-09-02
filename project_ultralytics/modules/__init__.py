"""Project-specific neural modules built on clean Ultralytics primitives."""

from .evidence import AugmentationAwareEvidence, GradientIsolatedEvidence, ScaleDisappearanceEvidence
from .fusion import WeightedAdd
from .attention import (
    FullSelfAttention,
    GlobalChannelContextCalibration,
    KVCompressedAttention,
    KVCompressedAttentionPartial,
    KVCompressedTransformerEncoder,
    PatchKVCompressedAttention,
    ReceptanceKVCompressedAttention,
    SurgicalPartialKVCompressedAttention,
    TopKAdaptiveGroupKVAttention,
    TopKGlobalGroupKVAttention,
)
from .cbam import CBAM, ChannelAttention, SpatialAttention
from .cbam_blocks import C2fCBAM, C3CBAM
from .feature_calibration import (
    AmplitudePerturbation,
    LearnableGlobalScalar,
    MatchedChannelPerturbation,
    P2AmplitudeCalibrator,
    P2FeatureProbe,
    ResidualDWConv,
    ResidualDWConv5,
)
from .dual_stream import DualChannelFormationBackbone, DualCollapse, DualDownsample
from .gt_cue_loss import (
    DedicatedCueSlots,
    DetachedResidualFusion,
    GTChannelSpecialization,
    GTCuePreservationHead,
    SplitChannelDetect,
)
from .input_cue import InputCueBank, InputCueConv, cue_channels
from .local_contrast import LocalContrastBasisStem, SidecarResidualFusionStem, SingleContrastFormationStem
from .nat import C2fNAT, NATBlock
from .routing_attention import BiLevelRoutingAttention
from .pconv import C2f_PConv, FasterNetBlock, PConv
from .raw_cue_fusion import MultiCueEvidenceFusion, RawColorSlotFusion, RawImageCueBank

__all__ = (
    "AugmentationAwareEvidence",
    "WeightedAdd",
    "AmplitudePerturbation",
    "BiLevelRoutingAttention",
    "CBAM",
    "C2fCBAM",
    "C3CBAM",
    "ChannelAttention",
    "C2f_PConv",
    "C2fNAT",
    "DedicatedCueSlots",
    "DualChannelFormationBackbone",
    "DualCollapse",
    "DualDownsample",
    "DetachedResidualFusion",
    "FasterNetBlock",
    "FullSelfAttention",
    "GlobalChannelContextCalibration",
    "GTChannelSpecialization",
    "GTCuePreservationHead",
    "GradientIsolatedEvidence",
    "InputCueBank",
    "InputCueConv",
    "KVCompressedAttention",
    "KVCompressedAttentionPartial",
    "KVCompressedTransformerEncoder",
    "LearnableGlobalScalar",
    "LocalContrastBasisStem",
    "MultiCueEvidenceFusion",
    "MatchedChannelPerturbation",
    "P2AmplitudeCalibrator",
    "P2FeatureProbe",
    "PConv",
    "NATBlock",
    "PatchKVCompressedAttention",
    "RawColorSlotFusion",
    "RawImageCueBank",
    "ReceptanceKVCompressedAttention",
    "ResidualDWConv",
    "ResidualDWConv5",
    "ScaleDisappearanceEvidence",
    "SidecarResidualFusionStem",
    "SingleContrastFormationStem",
    "SplitChannelDetect",
    "SpatialAttention",
    "SurgicalPartialKVCompressedAttention",
    "TopKAdaptiveGroupKVAttention",
    "TopKGlobalGroupKVAttention",
)
