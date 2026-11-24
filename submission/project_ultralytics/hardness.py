"""Detached hardness statistics shared by OACP-compatible loss paths."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def foreground_assignment_hardness(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    fg_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-image BCE over assigned foreground anchors only.

    Images without a positive assignment return NaN. The trainer ignores NaN
    values instead of interpreting no assignment as an easy image.
    """
    per_anchor = F.binary_cross_entropy_with_logits(
        pred_scores.detach(), target_scores.detach().to(pred_scores.dtype), reduction="none"
    ).mean(-1)
    result = torch.full(
        (pred_scores.shape[0],), float("nan"), device=pred_scores.device, dtype=per_anchor.dtype
    )
    counts = fg_mask.sum(1)
    valid = counts > 0
    if valid.any():
        result[valid] = (
            per_anchor[valid] * fg_mask[valid].to(per_anchor.dtype)
        ).sum(1) / counts[valid].to(per_anchor.dtype)
    return result
