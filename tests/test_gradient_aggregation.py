from __future__ import annotations

import torch

from project_ultralytics.gradient_aggregation import mode_balanced_gradient


def test_mode_balanced_gradient_equalizes_direction_modes() -> None:
    gradients = torch.tensor(
        [
            [10.0, 0.0],
            [8.0, 0.0],
            [0.0, 1.0],
        ]
    )

    aggregate, assignments = mode_balanced_gradient(gradients, n_modes=2)

    assert aggregate.shape == gradients.shape[1:]
    assert assignments.shape == (3,)
    assert assignments.unique().numel() == 2
    # Mode 0 has two objects, mode 1 has one. Equal mode weighting gives the
    # small orthogonal mode a visible contribution instead of a sample-count
    # weighted mean dominated by the first two objects.
    assert torch.allclose(aggregate, torch.tensor([4.5, 0.5]), atol=1e-5)


def test_mode_balanced_gradient_preserves_single_object_mean() -> None:
    gradients = torch.randn(1, 2, 3)

    aggregate, assignments = mode_balanced_gradient(gradients, n_modes=2)

    assert torch.equal(assignments, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(aggregate, gradients[0])
