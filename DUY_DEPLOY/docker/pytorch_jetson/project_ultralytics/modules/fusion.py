"""Project-owned feature fusion modules."""

from __future__ import annotations

import torch
from torch import nn


class WeightedAdd(nn.Module):
    """Fuse an arbitrary number of aligned feature maps with positive weights.

    Historical YAMLs use both two-way and three-way fusions while leaving the
    constructor arguments empty. The input arity is therefore inferred on the
    first forward pass instead of being hard-coded to two branches.
    """

    def __init__(self, n_inputs: int | None = None, eps: float = 1e-4) -> None:
        super().__init__()
        self.n_inputs = int(n_inputs) if n_inputs is not None else None
        if self.n_inputs is not None and self.n_inputs < 1:
            raise ValueError("n_inputs must be positive")
        self.eps = eps
        self.weights: nn.Parameter | None = None
        if self.n_inputs is not None:
            self.weights = nn.Parameter(torch.ones(self.n_inputs, dtype=torch.float32))

    def _ensure_weights(self, n_inputs: int, device: torch.device) -> nn.Parameter:
        if n_inputs < 1:
            raise ValueError("WeightedAdd requires at least one input")
        if self.weights is None:
            self.n_inputs = n_inputs
            self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32, device=device))
        elif self.weights.numel() != n_inputs:
            raise ValueError(f"WeightedAdd configured for {self.weights.numel()} inputs but received {n_inputs}")
        return self.weights

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return the weighted sum, preserving the feature shape."""
        if not isinstance(inputs, (list, tuple)):
            raise TypeError("WeightedAdd expects a list or tuple of feature maps")
        weights = torch.relu(self._ensure_weights(len(inputs), inputs[0].device))
        weights = weights / (weights.sum() + self.eps)
        return sum(feature * weights[i].to(dtype=feature.dtype) for i, feature in enumerate(inputs))
