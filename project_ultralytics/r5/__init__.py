"""AS-NCCP R5 shared framework primitives."""
from .diagnostics import frequency_probabilities, r5_diagnostics, sampling_entropy
from .difficulty_controller import ScaleDifficultyController
from .samplers import ScaleBinSampler
from .online_tal import R5EpochFeedbackAdapter, TALDifficultyBatch, TALDifficultyCollector, attach_r5_feedback
from .scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec
from .shared_state import SharedProbabilityState

__all__ = [
    "DEFAULT_SCALE_BINS",
    "R5EpochFeedbackAdapter",
    "TALDifficultyBatch",
    "TALDifficultyCollector",
    "attach_r5_feedback",
    "ScaleBinSampler",
    "ScaleBinSpec",
    "ScaleDifficultyController",
    "SharedProbabilityState",
    "frequency_probabilities",
    "r5_diagnostics",
    "sampling_entropy",
]
