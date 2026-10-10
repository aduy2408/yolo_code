"""Adaptive scale-aware negative-canvas Copy-Paste integration."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

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
        **kwargs,
    ) -> None:
        if scale_sampler is not None and scale_sampler.bin_spec != bin_spec:
            raise ValueError("scale_sampler and bin_spec must use the same scale bins")
        self.bin_spec = bin_spec
        self.scale_sampler = scale_sampler or ScaleBinSampler(bin_spec=bin_spec, rng=kwargs.get("rng"))
        super().__init__(dataset=dataset, target_policy="empirical", **kwargs)
        self.scale_bins = Counter()
        self.scale_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}

    def _build_pool(self) -> None:
        if self._study_pool_built:
            return
        super()._build_pool()
        self.scale_bins.clear()
        self.scale_bin_values = {bin_id: [] for bin_id in self.bin_spec.bin_ids}
        for size in self.target_sizes:
            bin_id = self.bin_spec.index(size)
            if bin_id is None:
                continue
            self.scale_bins[bin_id] += 1
            self.scale_bin_values[bin_id].append(float(size))

    def _sample_target_size(self, labels: dict[str, Any] | None = None) -> float:
        self._build_pool()
        return self.scale_sampler.sample(self.scale_bin_values)

    def update_probabilities(self, probabilities: Mapping[int, float] | list[float]) -> None:
        if isinstance(probabilities, Mapping):
            values = [probabilities.get(bin_id, 0.0) for bin_id in self.bin_spec.bin_ids]
        else:
            values = probabilities
        self.scale_sampler.set_probabilities(values)

    def r5_state_dict(self) -> dict[str, object]:
        return {
            "bin_spec": self.bin_spec.to_dict(),
            "scale_sampler": self.scale_sampler.state_dict(),
            "target_policy": "adaptive",
        }


__all__ = ["AdaptiveNegativeCanvasCopyPaste"]
