"""Adaptive scale-aware negative-canvas Copy-Paste integration."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np

from .negative_canvas_copy_paste import NegativeCanvasCopyPaste
from .r5.samplers import ScaleBinSampler
from .r5.scale_bins import DEFAULT_SCALE_BINS, ScaleBinSpec


class AdaptiveNegativeCanvasCopyPaste(NegativeCanvasCopyPaste):
    """R2-compatible transform whose only changing factor is target-scale sampling.

    Donor matching, hard paste, collision-aware placement, one pasted object,
    negative-canvas gating, and the default max-trial budget are inherited from
    :class:`NegativeCanvasCopyPaste`.
    """

    def __init__(
        self,
        dataset,
        scale_sampler: ScaleBinSampler | None = None,
        bin_spec: ScaleBinSpec = DEFAULT_SCALE_BINS,
        seed: int | None = None,
        worker_id: int = 0,
        **kwargs,
    ) -> None:
        if scale_sampler is not None and scale_sampler.bin_spec != bin_spec:
            raise ValueError("scale_sampler and bin_spec must use the same scale bins")
        self.bin_spec = bin_spec
        self.scale_sampler = scale_sampler or ScaleBinSampler(
            bin_spec=bin_spec,
            rng=kwargs.get("rng"),
            seed=seed,
            worker_id=worker_id,
        )
        super().__init__(dataset=dataset, target_policy="empirical", **kwargs)
        self.r5_provenance_enabled = True
        self.scale_bins = Counter()
        self.scale_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}
        self.donor_feasible_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}

    def _build_pool(self) -> None:
        if self._study_pool_built:
            return
        super()._build_pool()
        self.scale_bins.clear()
        self.scale_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}
        self.donor_feasible_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}
        donor_sizes = [self._effective_source_size(record) for record in self.object_pool]
        for size in self.target_sizes:
            bin_id = self.bin_spec.index(size)
            if bin_id is None:
                continue
            value = float(size)
            self.scale_bins[bin_id] += 1
            self.scale_bin_values[bin_id].append(value)
            if any(self._donor_valid(source_size, value) for source_size in donor_sizes):
                self.donor_feasible_bin_values[bin_id].append(value)

    def _sample_target_size(self, labels: dict[str, Any] | None = None) -> float:
        self._build_pool()
        return self.scale_sampler.sample(self.donor_feasible_bin_values)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Ensure every R5 sample carries an aligned original-GT mask."""
        if "r5_original_gt_mask" not in labels:
            labels["r5_original_gt_mask"] = np.ones(
                len(labels.get("instances", ())),
                dtype=bool,
            )
        return super().__call__(labels)

    def update_probabilities(self, probabilities: Mapping[int, float] | list[float]) -> None:
        if isinstance(probabilities, Mapping):
            values = [probabilities.get(bin_id, 0.0) for bin_id in self.bin_spec.bin_ids]
        else:
            values = probabilities
        self.scale_sampler.set_probabilities(values)

    def update_from_controller(self, controller, frequency=None, hybrid_ratio: float = 1.0) -> list[float]:
        """Apply an epoch-level controller distribution to this transform."""
        self._build_pool()
        feasible = self.scale_sampler.feasible_mask(self.donor_feasible_bin_values)
        probabilities = controller.probabilities(
            frequency=frequency,
            hybrid_ratio=hybrid_ratio,
            feasible=feasible,
        )
        self.update_probabilities(probabilities.tolist())
        return probabilities.tolist()

    def end_epoch_from_controller(self, controller, frequency=None, hybrid_ratio: float = 1.0) -> list[float]:
        controller.end_epoch()
        return self.update_from_controller(controller, frequency=frequency, hybrid_ratio=hybrid_ratio)

    def r5_state_dict(self) -> dict[str, object]:
        return {
            "bin_spec": self.bin_spec.to_dict(),
            "scale_sampler": self.scale_sampler.state_dict(),
            "target_policy": "adaptive",
        }


__all__ = ["AdaptiveNegativeCanvasCopyPaste"]
