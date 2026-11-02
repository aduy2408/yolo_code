import pytest
import torch

pytest.importorskip("cv2")

from ultralytics.nn.modules import ObjectRelativeFeatureSupervisor
from ultralytics.utils.loss import v8DetectionLoss


def _criterion(core_ratio=0.7):
    criterion = object.__new__(v8DetectionLoss)
    criterion.orfs_core_ratio = core_ratio
    criterion.orfs_center_gain = 1.0
    criterion.orfs_geometry_gain = 0.25
    criterion.p2_detail_metrics = {}
    return criterion


def _batch(boxes, batch_idx=None):
    return {
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "batch_idx": torch.zeros(len(boxes), dtype=torch.long) if batch_idx is None else torch.tensor(batch_idx),
    }


def test_orfs_is_training_only_passthrough_and_has_five_channels():
    module = ObjectRelativeFeatureSupervisor(16, hidden=8)
    x = torch.randn(2, 16, 8, 10)
    module.train()
    assert torch.equal(module(x), x)
    assert module.last_aux["orfs_structure_pred"].shape == (2, 5, 8, 10)
    module.eval()
    assert torch.equal(module(x), x)
    assert module.last_aux is None


def test_orfs_targets_are_resolution_consistent():
    criterion = _criterion()
    batch = _batch([[0.5, 0.5, 0.25, 0.5]])
    _, geo_a, mask_a = criterion._orfs_targets((32, 32), batch, 1, torch.float32)
    _, geo_b, mask_b = criterion._orfs_targets((64, 64), batch, 1, torch.float32)
    assert torch.allclose(geo_a[0, :, mask_a[0]].mean(1), geo_b[0, :, mask_b[0]].mean(1), atol=0.05)


def test_orfs_empty_image_loss_is_finite_and_backpropagates():
    criterion = _criterion()
    pred = torch.randn(1, 5, 8, 8, requires_grad=True)
    loss = criterion._orfs_structure_loss(
        {"boxes": pred.sum().reshape(1, 1), "p2_detail_aux": [{"orfs_structure_pred": pred}]},
        _batch([]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_orfs_overlap_uses_max_center_and_core_assignment():
    criterion = _criterion()
    batch = _batch([[0.45, 0.5, 0.3, 0.3], [0.55, 0.5, 0.3, 0.3]])
    center, _, mask = criterion._orfs_targets((16, 16), batch, 1, torch.float32)
    assert center.max() <= 1.0
    assert mask.any()
