import torch

from ultralytics.utils.loss import v8DetectionLoss


def test_scale_tempered_targets_positive_p2_only():
    loss = object.__new__(v8DetectionLoss)
    loss.scale_temper_lambda = 0.5
    loss.scale_temper_s1 = 16.0
    loss.scale_temper_s2 = 32.0
    loss.scale_temper_tau_min = 0.5
    loss.scale_temper_warmup_start = 5
    loss.scale_temper_warmup_end = 15
    loss.scale_temper_p2_only = True

    target_scores = torch.tensor([[[0.40], [0.00], [0.40]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 40.0, 40.0]]])
    target_gt_idx = torch.tensor([[0, 0, 1]])
    fg_mask = torch.tensor([[True, True, True]])

    loss.epoch = 0
    torch.testing.assert_close(loss.scale_tempered_cls_targets(target_scores, gt_bboxes, target_gt_idx, fg_mask, 2), target_scores)

    loss.epoch = 10
    out = loss.scale_tempered_cls_targets(target_scores, gt_bboxes, target_gt_idx, fg_mask, 2)
    assert out[0, 1, 0] == 0
    assert out[0, 2, 0] == target_scores[0, 2, 0]
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(0.45811388), rtol=1e-6, atol=1e-6)

    loss.epoch = 15
    out = loss.scale_tempered_cls_targets(target_scores, gt_bboxes, target_gt_idx, fg_mask, 2)
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(0.5162278), rtol=1e-6, atol=1e-6)
