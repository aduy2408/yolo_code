import pytest
import torch

from project_ultralytics.modules import WeightedAdd


@pytest.mark.parametrize("n_inputs", [2, 3])
def test_weighted_add_infers_yaml_fusion_arity(n_inputs: int) -> None:
    features = [torch.full((1, 4, 8, 8), float(i + 1)) for i in range(n_inputs)]
    module = WeightedAdd()
    output = module(features)
    assert output.shape == features[0].shape
    assert torch.isfinite(output).all()
    assert module.weights is not None
    assert module.weights.numel() == n_inputs


def test_weighted_add_rejects_mismatched_explicit_arity() -> None:
    with pytest.raises(ValueError, match="configured for 2 inputs"):
        WeightedAdd(2)([torch.ones(1, 1, 2, 2)] * 3)
