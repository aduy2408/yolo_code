import torch

from ultralytics.utils.loss import v8DetectionLoss


def test_factorized_tal_targets_sharpen_p2_small_gt_only():
    loss = object.__new__(v8DetectionLoss)
    loss.factorized_tal_lambda = 0.5
    loss.factorized_tal_tau = 0.75
    loss.factorized_tal_kappa = 1.5
    loss.factorized_tal_s_max = 32.0
    loss.factorized_tal_warmup_start = 5
    loss.factorized_tal_warmup_end = 15
    loss.factorized_tal_p2_only = True

    target_scores = torch.tensor([[[0.75], [0.49], [0.23], [0.00], [0.75], [0.40]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 40.0, 40.0]]])
    target_gt_idx = torch.tensor([[0, 0, 0, 0, 0, 1]])
    fg_mask = torch.tensor([[True, True, True, True, True, True]])
    pred_bboxes = gt_bboxes[0, target_gt_idx[0]].unsqueeze(0).clone()
    stride_tensor = torch.ones(6)

    loss.epoch = 0
    torch.testing.assert_close(
        loss.factorized_tal_cls_targets(
            target_scores, gt_bboxes, target_gt_idx, fg_mask, 4, pred_bboxes, stride_tensor
        ),
        target_scores,
    )

    loss.epoch = 15
    out = loss.factorized_tal_cls_targets(
        target_scores, gt_bboxes, target_gt_idx, fg_mask, 4, pred_bboxes, stride_tensor
    )
    assert out[0, 0, 0] > target_scores[0, 0, 0]
    assert out[0, 1, 0] < target_scores[0, 1, 0]
    assert out[0, 2, 0] < target_scores[0, 2, 0]
    assert out[0, 3, 0] == 0
    assert out[0, 4, 0] == target_scores[0, 4, 0]
    assert out[0, 5, 0] == target_scores[0, 5, 0]
    torch.testing.assert_close(target_scores.sum(), torch.tensor(2.62))
