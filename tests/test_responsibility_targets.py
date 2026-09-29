from __future__ import annotations

import torch

from ultralytics.utils.loss import (
    build_bounded_responsibility_target_scores,
    build_kl_responsibility_target_scores,
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


def _kl_fixture() -> tuple[torch.Tensor, ...]:
    target_scores = torch.zeros(1, 3, 1)
    target_scores[0, :, 0] = torch.tensor([0.2, 0.5, 0.3])
    target_gt_idx = torch.tensor([[0, 0, 0]])
    fg_mask = torch.ones(1, 3, dtype=torch.bool)
    gt_bboxes = torch.tensor([[[0.0, 0.0, 8.0, 8.0]]])
    pred_bboxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 4.0, 4.0], [2.0, 2.0, 8.0, 8.0]]]
    )
    stride_tensor = torch.ones(3, 1)
    return target_scores, target_gt_idx, fg_mask, gt_bboxes, pred_bboxes, stride_tensor


def test_kl_eta_zero_is_exact_identity() -> None:
    fixture = _kl_fixture()
    result, diagnostics = build_kl_responsibility_target_scores(
        *fixture, eta_max=0.0, tiny_max_dim=16.0
    )
    assert torch.equal(result, fixture[0])
    assert diagnostics["n_refined_gt"] == 0.0


def test_kl_preserves_per_gt_mass_and_favors_higher_iou() -> None:
    fixture = _kl_fixture()
    result, diagnostics = build_kl_responsibility_target_scores(
        *fixture, eta_max=0.25, tiny_max_dim=16.0
    )
    assert torch.allclose(result.sum(), fixture[0].sum(), atol=1e-6)
    assert result[0, 0, 0] > fixture[0][0, 0, 0]
    assert result[0, 1, 0] < fixture[0][0, 1, 0]
    assert diagnostics["mean_expected_utility_gain"] >= -1e-7
    assert diagnostics["max_mass_error"] < 1e-6


def test_kl_preserves_zero_tal_support() -> None:
    target_scores, target_gt_idx, fg_mask, gt_bboxes, pred_bboxes, stride_tensor = _kl_fixture()
    target_scores[0, 2, 0] = 0.0
    pred_bboxes[0, 2] = gt_bboxes[0, 0]
    result, _ = build_kl_responsibility_target_scores(
        target_scores, target_gt_idx, fg_mask, gt_bboxes, pred_bboxes, stride_tensor, eta_max=1.0, tiny_max_dim=16.0
    )
    assert result[0, 2, 0].item() == 0.0


def test_kl_large_gt_gate_is_exact_identity() -> None:
    fixture = _kl_fixture()
    fixture = (*fixture[:3], torch.tensor([[[0.0, 0.0, 64.0, 64.0]]]), *fixture[4:])
    result, _ = build_kl_responsibility_target_scores(
        *fixture, eta_max=1.0, tiny_max_dim=16.0
    )
    assert torch.equal(result, fixture[0])
