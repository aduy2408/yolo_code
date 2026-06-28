# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from models_related.ultralytics.ultralytics.cfg import get_cfg
from models_related.ultralytics.ultralytics.utils import DEFAULT_CFG
from models_related.ultralytics.ultralytics.utils.nms import TorchNMS, non_max_suppression


def test_torchnms_soft_nms_linear_keeps_decayed_box():
    """Test linear Soft-NMS keeps an overlapping box when its decayed score remains above min_score."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
    hard_scores = torch.tensor([0.9, 0.8, 0.7])
    soft_scores = hard_scores.clone()

    hard_keep = TorchNMS.nms(boxes, hard_scores, 0.5)
    soft_keep = TorchNMS.soft_nms(boxes, soft_scores, 0.5, method="linear", min_score=0.001)

    assert hard_keep.tolist() == [0, 2]
    assert soft_keep.tolist() == [0, 2, 1]
    assert soft_scores[1] < hard_scores[1]


def test_torchnms_soft_nms_gaussian_is_deterministic():
    """Test Gaussian Soft-NMS produces stable valid indices."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep = TorchNMS.soft_nms(boxes, scores, 0.5, method="gaussian", sigma=0.5, min_score=0.001)

    assert keep.tolist() == [0, 2, 1]
    assert set(keep.tolist()) == {0, 1, 2}


def test_non_max_suppression_soft_linear_keeps_more_boxes_than_hard():
    """Test Soft-NMS keeps more high-overlap detections than default hard NMS."""
    prediction = torch.zeros((1, 5, 3))
    prediction[0, :4] = torch.tensor(
        [[5.0, 6.0, 25.0], [5.0, 6.0, 25.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    )
    prediction[0, 4] = torch.tensor([0.9, 0.8, 0.7])

    hard = non_max_suppression(prediction.clone(), conf_thres=0.001, iou_thres=0.5, nc=1)
    soft = non_max_suppression(
        prediction.clone(),
        conf_thres=0.001,
        iou_thres=0.5,
        nc=1,
        nms_method="soft-linear",
        soft_nms_min_score=0.001,
    )
    limited = non_max_suppression(
        prediction.clone(),
        conf_thres=0.001,
        iou_thres=0.5,
        nc=1,
        max_det=2,
        nms_method="soft-linear",
        soft_nms_min_score=0.001,
    )

    assert len(hard[0]) == 2
    assert len(soft[0]) == 3
    assert len(limited[0]) == 2
    assert soft[0][:, 4].tolist() == pytest.approx([0.9, 0.7, 0.2555], abs=1e-4)


def test_non_max_suppression_soft_nms_return_idxs():
    """Test return_idxs maps Soft-NMS results back to original prediction indices."""
    prediction = torch.zeros((1, 5, 3))
    prediction[0, :4] = torch.tensor(
        [[5.0, 6.0, 25.0], [5.0, 6.0, 25.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    )
    prediction[0, 4] = torch.tensor([0.9, 0.8, 0.7])

    output, keep_idxs = non_max_suppression(
        prediction,
        conf_thres=0.001,
        iou_thres=0.5,
        nc=1,
        nms_method="soft-linear",
        soft_nms_min_score=0.001,
        return_idxs=True,
    )

    assert len(output[0]) == 3
    assert keep_idxs[0].tolist() == [0, 2, 1]


def test_nms_method_config_validation():
    """Test valid and invalid NMS method config overrides."""
    cfg = get_cfg(DEFAULT_CFG, {"nms_method": "soft-gaussian"})
    assert cfg.nms_method == "soft-gaussian"

    with pytest.raises(ValueError, match="nms_method=foo"):
        get_cfg(DEFAULT_CFG, {"nms_method": "foo"})
