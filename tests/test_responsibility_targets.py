from __future__ import annotations

import torch

from ultralytics.utils.loss import build_responsibility_target_scores


def test_iou_responsibility_preserves_gt_budget_and_changes_relative_weights() -> None:
    target_scores = torch.zeros(1, 3, 1)
    target_scores[0, :, 0] = 1
    target_gt_idx = torch.tensor([[0, 0, 0]])
    fg_mask = torch.ones(1, 3, dtype=torch.bool)
    gt_labels = torch.zeros(1, 1, 1)
    overlaps = torch.tensor([[[0.2, 0.5, 0.3]]])

    result = build_responsibility_target_scores(
        target_scores, target_gt_idx, fg_mask, gt_labels, overlaps, "iou", budget=1.0
    )

    assert torch.allclose(result.sum(), torch.tensor(1.0))
    assert torch.allclose(result[0, :, 0], torch.tensor([0.2, 0.5, 0.3]))


def test_iou_sqrt_softens_tiny_quality_gaps() -> None:
    target_scores = torch.ones(1, 2, 1)
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.ones(1, 2, dtype=torch.bool)
    gt_labels = torch.zeros(1, 1, 1)
    overlaps = torch.tensor([[[0.01, 0.25]]])

    result = build_responsibility_target_scores(
        target_scores, target_gt_idx, fg_mask, gt_labels, overlaps, "iou_sqrt", budget=1.0
    )

    assert result[0, 0, 0] > 0
    assert result[0, 1, 0] > result[0, 0, 0]
    assert torch.allclose(result.sum(), torch.tensor(1.0))
