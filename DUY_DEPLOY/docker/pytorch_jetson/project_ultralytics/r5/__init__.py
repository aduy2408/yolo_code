"""AS-NCCP R5 shared framework primitives."""
from .diagnostics import frequency_probabilities, r5_diagnostics, sampling_entropy
from .difficulty_controller import ScaleDifficultyController
from .samplers import ScaleBinSampler
from .online_tal import (
    R5EpochFeedbackAdapter,
    TALDifficultyBatch,
    TALDifficultyCollector,
    attach_r5_feedback,
    register_r5_epoch_callback,
)
from .scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec
from .shared_state import SharedProbabilityState
from .recall_evaluator import RecallEvaluator, RecallProfile, box_iou, match_predictions
from .online_probe import ProbeResult, R5ProbeCallback
from .checkpoint import R5StateCheckpointCallback

__all__ = [
    "DEFAULT_SCALE_BINS",
    "R5EpochFeedbackAdapter",
    "TALDifficultyBatch",
    "TALDifficultyCollector",
    "attach_r5_feedback",
    "register_r5_epoch_callback",
    "ScaleBinSampler",
    "ScaleBinSpec",
    "ScaleDifficultyController",
    "SharedProbabilityState",
    "RecallEvaluator",
    "RecallProfile",
    "box_iou",
    "match_predictions",
    "ProbeResult",
    "R5ProbeCallback",
    "R5StateCheckpointCallback",
    "frequency_probabilities",
    "r5_diagnostics",
    "sampling_entropy",
]
