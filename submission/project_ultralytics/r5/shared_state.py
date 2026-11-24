"""Shared-memory epoch-boundary probability snapshots for R5 workers."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


class SharedProbabilityState:
    """Double-buffered CPU probabilities safe to read from DataLoader workers."""

    def __init__(self, num_bins: int, probabilities: Iterable[float] | None = None) -> None:
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        self.num_bins = int(num_bins)
        self._buffers = torch.zeros((2, self.num_bins), dtype=torch.float32).share_memory_()
        self._version = torch.zeros(1, dtype=torch.int64).share_memory_()
        initial = np.full(self.num_bins, 1.0 / self.num_bins) if probabilities is None else np.asarray(list(probabilities), dtype=np.float64)
        self.update(initial)

    def update(self, probabilities: Iterable[float]) -> None:
        values = np.asarray(list(probabilities), dtype=np.float64)
        if values.shape != (self.num_bins,):
            raise ValueError("probabilities must have one value per bin")
        if np.any(values < 0.0) or not np.all(np.isfinite(values)) or values.sum() <= 0.0:
            raise ValueError("probabilities must be finite, non-negative, and non-zero")
        values = values / values.sum()
        current = int(self._version.item())
        next_slot = (current + 1) % 2
        self._buffers[next_slot].copy_(torch.as_tensor(values, dtype=torch.float32))
        self._version.fill_(current + 1)

    def snapshot(self) -> np.ndarray:
        for _ in range(8):
            version_before = int(self._version.item())
            values = self._buffers[version_before % 2].clone().numpy()
            if version_before == int(self._version.item()):
                return values.astype(np.float64, copy=False)
        raise RuntimeError("shared R5 probability state changed during snapshot")

    def state_dict(self) -> dict[str, object]:
        return {
            "num_bins": self.num_bins,
            "probabilities": self.snapshot().tolist(),
            "version": int(self._version.item()),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["num_bins"]) != self.num_bins:
            raise ValueError("shared probability state has a different number of bins")
        self.update(state["probabilities"])

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "SharedProbabilityState":
        # ``__init__`` publishes the supplied probabilities into a coherent
        # buffer/version pair. The version is only a publication counter and
        # must not be restored independently of its corresponding buffer.
        return cls(int(state["num_bins"]), state["probabilities"])


__all__ = ["SharedProbabilityState"]
