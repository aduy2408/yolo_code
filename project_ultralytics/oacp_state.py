"""Shared state for worker-safe adaptive OACP policies.

The augmentation pipeline runs in persistent dataloader workers.  Plain Python
attributes updated by the trainer would be copied at worker creation and never
observe later epochs, so the small amount of state needed by OACP lives in
multiprocessing shared objects instead.
"""
from __future__ import annotations

from multiprocessing import Array, Value
from typing import Iterable


class OACPSharedState:
    """Worker-visible epoch and previous-epoch hardness state.

    The trainer owns the temporary per-epoch accumulators.  Workers only read
    ``epoch`` and the committed EMA hardness array.
    """

    def __init__(self, size: int, *, beta: float = 0.9) -> None:
        if int(size) < 0:
            raise ValueError("shared state size must be non-negative")
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("hardness EMA beta must be in [0, 1)")
        self.size = int(size)
        self.beta = float(beta)
        self.epoch = Value("i", 0)
        self.total_epochs = Value("i", 0)
        self._hardness = Array("d", [0.5] * self.size, lock=True)
        self._hardness_seen = Array("b", [0] * self.size, lock=True)
        self._epoch_sum: dict[int, float] = {}
        self._epoch_count: dict[int, int] = {}
        self.last_epoch_stats: dict[str, float] = {}

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.epoch.value = int(epoch)
        if total_epochs is not None:
            self.total_epochs.value = int(total_epochs)

    def begin_epoch(self) -> None:
        self._epoch_sum.clear()
        self._epoch_count.clear()

    def read_hardness(self, index: int) -> float:
        index = int(index)
        if index < 0 or index >= self.size:
            return 0.5
        with self._hardness.get_lock():
            return float(self._hardness[index])

    def read_hardness_many(self, indices: Iterable[int]) -> list[float]:
        return [self.read_hardness(index) for index in indices]

    def update_hardness(self, indices: Iterable[int], values: Iterable[float]) -> None:
        """Accumulate detached batch hardness in the trainer process."""
        for index, value in zip(indices, values, strict=False):
            index = int(index)
            if index < 0 or index >= self.size:
                continue
            value = float(value)
            if not value == value:  # NaN
                continue
            self._epoch_sum[index] = self._epoch_sum.get(index, 0.0) + value
            self._epoch_count[index] = self._epoch_count.get(index, 0) + 1

    def finish_epoch(self) -> dict[str, float]:
        """Normalize current-epoch values and commit them as next-epoch EMA."""
        if not self._epoch_sum:
            self.last_epoch_stats = {
                "hardness_updates": 0.0,
                "hardness_min": 0.0,
                "hardness_max": 0.0,
                "hardness_mean": 0.5,
                "hardness_observed_fraction": 0.0,
            }
            return dict(self.last_epoch_stats)
        raw = {
            index: self._epoch_sum[index] / max(self._epoch_count[index], 1)
            for index in self._epoch_sum
        }
        values = list(raw.values())
        low, high = min(values), max(values)
        span = max(high - low, 1e-8)
        normalized = {index: (value - low) / span for index, value in raw.items()}
        with self._hardness.get_lock():
            for index, value in normalized.items():
                self._hardness[index] = self.beta * self._hardness[index] + (1.0 - self.beta) * value
        with self._hardness_seen.get_lock():
            for index in normalized:
                self._hardness_seen[index] = 1
        self.last_epoch_stats = {
            "hardness_updates": float(len(normalized)),
            "hardness_min": float(low),
            "hardness_max": float(high),
            "hardness_mean": float(sum(normalized.values()) / len(normalized)),
            "hardness_observed_fraction": float(len(normalized) / max(self.size, 1)),
        }
        self._epoch_sum.clear()
        self._epoch_count.clear()
        return dict(self.last_epoch_stats)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "epoch": int(self.epoch.value),
            "total_epochs": int(self.total_epochs.value),
            **self.last_epoch_stats,
        }
