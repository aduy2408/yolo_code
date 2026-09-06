import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "models_related" / "ultralytics"))
from ultralytics.nn.modules import ComplementaryEvidenceFusion


@pytest.mark.parametrize("mode", ["predictable", "hard_negative", "retrieval"])
def test_complementary_evidence_is_identity_initialized_and_has_source_aux(mode):
    module = ComplementaryEvidenceFusion(32, 16, mode).train()
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    raw = torch.randn(2, 16, 8, 8)
    output = module((x, raw))
    assert torch.equal(output, x)
    batch = {
        "batch_idx": torch.tensor([0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, _ = module.auxiliary_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert module.source_selector[0].weight.grad is not None


def test_hard_negative_uses_detector_confidence_not_reconstruction_hardness():
    module = ComplementaryEvidenceFusion(32, 16, "hard_negative").train()
    x = torch.randn(1, 32, 4, 4, requires_grad=True)
    raw = torch.randn(1, 16, 4, 4)
    module((x, raw))
    batch = {"batch_idx": torch.tensor([0]), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]])}
    context = {
        "p2_fg_mask": torch.zeros(1, 16, dtype=torch.bool),
        "p2_pred_conf": torch.linspace(0.0, 1.0, 16).reshape(1, 16),
    }
    _, metrics = module.auxiliary_loss(batch, context)
    assert metrics["hard_negative_count"].item() == 2
