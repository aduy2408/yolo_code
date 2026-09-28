from __future__ import annotations

import torch

from ultralytics.utils.loss import (
    build_bounded_responsibility_target_scores,
    build_responsibility_target_scores,
)


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


def test_bounded_residual_preserves_tal_mass() -> None:
    target_scores = torch.zeros(1, 3, 1)
    target_scores[0, :, 0] = torch.tensor([0.2, 0.5, 0.3])
    target_gt_idx = torch.tensor([[0, 0, 0]])
    fg_mask = torch.ones(1, 3, dtype=torch.bool)
    gt_labels = torch.zeros(1, 1, 1)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
    overlaps = torch.tensor([[[0.2, 0.5, 0.3]]])

    result = build_bounded_responsibility_target_scores(
        target_scores,
        target_gt_idx,
        fg_mask,
        gt_labels,
        gt_bboxes,
        overlaps,
        "residual",
        lambda_max=0.25,
        clip=0.5,
    )

    assert torch.allclose(result.sum(), target_scores.sum())
    assert torch.all(result >= 0)
    assert not torch.allclose(result, target_scores)


def test_tiny_gate_disables_residual_for_large_gt() -> None:
    target_scores = torch.zeros(1, 2, 1)
    target_scores[0, :, 0] = torch.tensor([0.4, 0.6])
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.ones(1, 2, dtype=torch.bool)
    gt_labels = torch.zeros(1, 1, 1)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 64.0, 64.0]]])
    overlaps = torch.tensor([[[0.1, 0.9]]])

    result = build_bounded_responsibility_target_scores(
        target_scores,
        target_gt_idx,
        fg_mask,
        gt_labels,
        gt_bboxes,
        overlaps,
        "residual_tiny",
        lambda_max=0.25,
        clip=0.5,
    )

    assert torch.allclose(result, target_scores)


def test_curriculum_inherits_tiny_gate_for_large_gt() -> None:
    target_scores = torch.zeros(1, 2, 1)
    target_scores[0, :, 0] = torch.tensor([0.4, 0.6])
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.ones(1, 2, dtype=torch.bool)
    gt_labels = torch.zeros(1, 1, 1)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 64.0, 64.0]]])
    overlaps = torch.tensor([[[0.1, 0.9]]])

    result = build_bounded_responsibility_target_scores(
        target_scores,
        target_gt_idx,
        fg_mask,
        gt_labels,
        gt_bboxes,
        overlaps,
        "residual_curriculum",
        lambda_max=0.25,
        clip=0.5,
    )

    assert torch.allclose(result, target_scores)
