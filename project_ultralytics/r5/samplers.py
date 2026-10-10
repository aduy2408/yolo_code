"""Scale samplers shared by R5 negative-canvas transforms."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec


class ScaleBinSampler:
    """Sample a target size from controller probabilities and feasible values."""

    def __init__(
        self,
        bin_spec: ScaleBinSpec = DEFAULT_SCALE_BINS,
        rng=None,
        probabilities: Iterable[float] | None = None,
    ) -> None:
        self.bin_spec = bin_spec
        self.rng = rng
        self._numpy_rng = np.random.default_rng()
        self._probabilities = np.full(bin_spec.num_bins, 1.0 / bin_spec.num_bins, dtype=np.float64)
        if probabilities is not None:
            self.set_probabilities(probabilities)

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    def set_probabilities(self, probabilities: Iterable[float]) -> None:
        values = np.asarray(list(probabilities), dtype=np.float64)
        if values.shape != (self.bin_spec.num_bins,):
            raise ValueError("probabilities must have one value per scale bin")
        if np.any(values < 0.0) or not np.all(np.isfinite(values)) or values.sum() <= 0.0:
            raise ValueError("probabilities must be finite, non-negative, and non-zero")
        self._probabilities = values / values.sum()

    def feasible_mask(self, values_by_bin: Mapping[int, Iterable[float]]) -> np.ndarray:
        return np.asarray([bool(list(values_by_bin.get(bin_id, ()))) for bin_id in self.bin_spec.bin_ids], dtype=bool)

    def sample(self, values_by_bin: Mapping[int, Iterable[float]]) -> float:
        normalized = {int(bin_id): [float(value) for value in values] for bin_id, values in values_by_bin.items()}
        feasible = self.feasible_mask(normalized)
        if not np.any(feasible):
            raise ValueError("no feasible target scale bins")
        probabilities = self._probabilities.copy()
        probabilities[~feasible] = 0.0
        if probabilities.sum() <= 0.0:
            probabilities = feasible.astype(np.float64)
        probabilities /= probabilities.sum()
        ids = self.bin_spec.bin_ids
        if self.rng is None:
            selected = int(self._numpy_rng.choice(ids, p=probabilities))
            return float(self._numpy_rng.choice(normalized[selected]))
        selected = self.rng.choices(ids, weights=probabilities.tolist(), k=1)[0]
        return float(self.rng.choice(normalized[selected]))

    def state_dict(self) -> dict[str, object]:
        return {"edges": list(self.bin_spec.edges), "probabilities": self._probabilities.tolist()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object], rng=None) -> "ScaleBinSampler":
        sampler = cls(ScaleBinSpec.from_edges(state["edges"]), rng=rng)
        sampler.set_probabilities(state["probabilities"])
        return sampler


__all__ = ["ScaleBinSampler"]
