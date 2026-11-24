"""Diagnostics helpers for R5 sampling and estimator feedback."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sampling_entropy(probabilities: Iterable[float], epsilon: float = 1e-12) -> float:
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size == 0 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= epsilon:
        return 0.0
    values = values / total
    return float(-(values[values > epsilon] * np.log(values[values > epsilon])).sum())


def frequency_probabilities(counts: Iterable[float], eta: float = 1.0, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(list(counts), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("counts must be a non-empty vector")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("counts must be finite and non-negative")
    if eta < 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and non-negative")
    weights = np.power(values + epsilon, -eta)
    return weights / weights.sum()


def r5_diagnostics(
    proposed_probabilities: Iterable[float],
    attempted_pastes: Iterable[float],
    successful_pastes: Iterable[float],
    difficulty: Iterable[float] | None = None,
    donor_failed: Iterable[float] | None = None,
    placement_failed: Iterable[float] | None = None,
    crop_failed: Iterable[float] | None = None,
) -> dict[str, object]:
    proposed = np.asarray(list(proposed_probabilities), dtype=np.float64)
    attempted = np.asarray(list(attempted_pastes), dtype=np.float64)
    successful = np.asarray(list(successful_pastes), dtype=np.float64)
    if not (proposed.shape == attempted.shape == successful.shape):
        raise ValueError("all per-bin diagnostics must have the same shape")
    for name, values in (("proposed_probabilities", proposed), ("attempted_pastes", attempted), ("successful_pastes", successful)):
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite and non-negative")
    actual = successful / max(float(successful.sum()), 1.0)
    result: dict[str, object] = {
        "probability": proposed.tolist(),
        "attempted_pastes": attempted.tolist(),
        "successful_pastes": successful.tolist(),
        "actual_scale_distribution": actual.tolist(),
        "sampling_entropy": sampling_entropy(proposed),
    }
    failures = {}
    for name, values in (("donor_failed", donor_failed), ("placement_failed", placement_failed), ("crop_failed", crop_failed)):
        if values is None:
            continue
        array = np.asarray(list(values), dtype=np.float64)
        if array.shape != proposed.shape or np.any(array < 0.0) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite, non-negative, and match the per-bin shape")
        result[name] = array.tolist()
        failures[name] = float(array.sum())
    if failures:
        result["total_failures"] = sum(failures.values())
    else:
        result["total_failures"] = float(max(attempted.sum() - successful.sum(), 0.0))
    if difficulty is not None:
        values = np.asarray(list(difficulty), dtype=np.float64)
        if values.shape != proposed.shape:
            raise ValueError("difficulty must have one value per bin")
        result["difficulty"] = values.tolist()
    return result


__all__ = ["frequency_probabilities", "r5_diagnostics", "sampling_entropy"]
