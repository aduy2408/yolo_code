"""Difficulty aggregation and probability control for AS-NCCP R5."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


@dataclass
class ScaleDifficultyController:
    """Aggregate per-object difficulty and derive stable scale probabilities.

    Aggregation is performed as sum/count over all observations. This avoids
    giving a tiny mini-batch the same influence as a large one. ``end_epoch``
    applies the EMA once, after optional distributed reduction by the caller.
    """

    num_bins: int
    beta: float = 0.9
    gamma: float = 1.0
    exploration: float = 0.2
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_bins < 1:
            raise ValueError("num_bins must be positive")
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if self.gamma < 0.0 or not math.isfinite(self.gamma):
            raise ValueError("gamma must be finite and non-negative")
        if not 0.0 <= self.exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        self.difficulty = np.full(self.num_bins, 0.5, dtype=np.float64)
        self.sum_difficulty = np.zeros(self.num_bins, dtype=np.float64)
        self.count = np.zeros(self.num_bins, dtype=np.int64)
        self.epoch = 0

    def reset_pending(self) -> None:
        self.sum_difficulty.fill(0.0)
        self.count.fill(0)

    def accumulate(self, bin_ids: Iterable[int], values: Iterable[float]) -> None:
        ids = np.asarray(list(bin_ids), dtype=np.int64)
        vals = np.asarray(list(values), dtype=np.float64)
        if ids.shape != vals.shape:
            raise ValueError("bin_ids and values must have the same length")
        if ids.size == 0:
            return
        if np.any(ids < 0) or np.any(ids >= self.num_bins):
            raise ValueError("bin id is outside controller range")
        if not np.all(np.isfinite(vals)) or np.any(vals < 0.0) or np.any(vals > 1.0):
            raise ValueError("difficulty values must be finite and in [0, 1]")
        np.add.at(self.sum_difficulty, ids, vals)
        np.add.at(self.count, ids, 1)

    def accumulate_mapping(self, aggregates: Mapping[int, tuple[float, int]]) -> None:
        for bin_id, (total, count) in aggregates.items():
            index = int(bin_id)
            if index < 0 or index >= self.num_bins:
                raise ValueError("aggregate bin id is outside controller range")
            if count < 0 or not math.isfinite(float(total)) or float(total) < 0.0 or float(total) > count + self.epsilon:
                raise ValueError("invalid difficulty aggregate")
            if count:
                self.sum_difficulty[index] += float(total)
                self.count[index] += int(count)

    def end_epoch(self) -> np.ndarray:
        observed = self.count > 0
        means = np.divide(
            self.sum_difficulty,
            np.maximum(self.count, 1),
            out=np.zeros_like(self.sum_difficulty),
            where=self.count > 0,
        )
        means = np.clip(means, 0.0, 1.0)
        self.difficulty[observed] = self.beta * self.difficulty[observed] + (1.0 - self.beta) * means[observed]
        self.reset_pending()
        self.epoch += 1
        return self.difficulty.copy()

    def probabilities(
        self,
        frequency: Iterable[float] | None = None,
        difficulty: Iterable[float] | None = None,
        hybrid_ratio: float = 1.0,
        feasible: Iterable[bool] | None = None,
    ) -> np.ndarray:
        """Return a normalized difficulty, frequency, or hybrid distribution."""
        if not 0.0 <= hybrid_ratio <= 1.0:
            raise ValueError("hybrid_ratio must be in [0, 1]")
        diff = np.asarray(self.difficulty if difficulty is None else list(difficulty), dtype=np.float64)
        if diff.shape != (self.num_bins,):
            raise ValueError("difficulty must have one value per bin")
        if np.any(~np.isfinite(diff)) or np.any(diff < 0.0) or np.any(diff > 1.0):
            raise ValueError("difficulty must be finite and in [0, 1]")
        diff = diff ** self.gamma
        diff = self._normalize(diff)
        if frequency is None:
            base = diff
        else:
            freq = np.asarray(list(frequency), dtype=np.float64)
            if freq.shape != (self.num_bins,):
                raise ValueError("frequency must have one value per bin")
            if np.any(freq < 0.0) or not np.all(np.isfinite(freq)):
                raise ValueError("frequency must be finite and non-negative")
            base = (1.0 - hybrid_ratio) * self._normalize(freq) + hybrid_ratio * diff
        feasible_mask = np.ones(self.num_bins, dtype=bool) if feasible is None else np.asarray(list(feasible), dtype=bool)
        if feasible_mask.shape != (self.num_bins,):
            raise ValueError("feasible must have one value per bin")
        if not np.any(feasible_mask):
            raise ValueError("at least one scale bin must be feasible")
        base = np.where(feasible_mask, base, 0.0)
        base = self._normalize(base, feasible_mask)
        exploration = np.where(feasible_mask, 1.0, 0.0)
        exploration = self._normalize(exploration, feasible_mask)
        return (1.0 - self.exploration) * base + self.exploration * exploration

    def state_dict(self) -> dict[str, object]:
        return {
            "num_bins": self.num_bins,
            "beta": self.beta,
            "gamma": self.gamma,
            "exploration": self.exploration,
            "epsilon": self.epsilon,
            "difficulty": self.difficulty.tolist(),
            "sum_difficulty": self.sum_difficulty.tolist(),
            "count": self.count.tolist(),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["num_bins"]) != self.num_bins:
            raise ValueError("controller state has a different number of bins")
        for name, target in (("difficulty", self.difficulty), ("sum_difficulty", self.sum_difficulty), ("count", self.count)):
            if name not in state:
                raise ValueError(f"controller state is missing {name!r}")
            values = np.asarray(state[name], dtype=target.dtype)
            if values.shape != target.shape:
                raise ValueError(f"controller state field {name!r} has the wrong shape")
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"controller state field {name!r} is invalid")
            target[...] = values
        if np.any(self.difficulty > 1.0) or np.any(self.sum_difficulty > self.count + self.epsilon):
            raise ValueError("controller state contains out-of-range difficulty")
        self.epoch = int(state.get("epoch", 0))
        if self.epoch < 0:
            raise ValueError("controller epoch must be non-negative")

    def _normalize(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        total = float(values.sum())
        if total <= self.epsilon:
            allowed = np.ones(self.num_bins, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
            if not np.any(allowed):
                raise ValueError("normalization mask has no active bins")
            result = np.zeros(self.num_bins, dtype=np.float64)
            result[allowed] = 1.0 / allowed.sum()
            return result
        return values / total


__all__ = ["ScaleDifficultyController"]
