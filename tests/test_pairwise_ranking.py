from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss


def _criterion(mode: str) -> v8DetectionLoss:
    criterion = object.__new__(v8DetectionLoss)
    criterion.rank_mode = mode
    criterion.rank_topk = 10
    criterion.rank_tau = 0.25
    criterion.rank_iou_margin = 0.10
    criterion.rank_teacher_margin = 0.05
    criterion.rank_lambda_loc = 0.25
    criterion.rank_tiny_max_dim = 16.0
    return criterion


def _inputs(gt_size: float = 8.0):
    pred_scores = torch.tensor([[[0.1], [0.4], [0.2]]], requires_grad=True)
    pred_bboxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 4.0, 4.0], [2.0, 2.0, 8.0, 8.0]]]
    )
    target_bboxes = torch.tensor(
        [[[0.0, 0.0, gt_size, gt_size], [0.0, 0.0, gt_size, gt_size], [0.0, 0.0, gt_size, gt_size]]]
    )
    target_scores = torch.ones(1, 3, 1)
    target_gt_idx = torch.zeros(1, 3, dtype=torch.long)
    fg_mask = torch.ones(1, 3, dtype=torch.bool)
    stride_tensor = torch.ones(3, 1)
    gt_bboxes = torch.tensor([[[0.0, 0.0, gt_size, gt_size]]])
    return pred_scores, pred_bboxes, target_bboxes, target_scores, target_gt_idx, fg_mask, stride_tensor, gt_bboxes


def test_r1_localization_ranking_has_gradient_only_for_tiny_gt() -> None:
    criterion = _criterion("localization")
    inputs = _inputs()
    loss = criterion.pairwise_ranking_loss(*inputs)
    assert loss.grad_fn is not None
    loss.backward()
    assert inputs[0].grad is not None


def test_j1_joint_ranking_is_finite_and_large_gt_is_identity() -> None:
    criterion = _criterion("joint")
    tiny = criterion.pairwise_ranking_loss(*_inputs())
    large = criterion.pairwise_ranking_loss(*_inputs(64.0))
    assert torch.isfinite(tiny)
    assert large.item() == 0.0


def test_ranking_mode_off_is_exact_zero() -> None:
    criterion = _criterion("off")
    loss = criterion.pairwise_ranking_loss(*_inputs())
    assert loss.item() == 0.0
