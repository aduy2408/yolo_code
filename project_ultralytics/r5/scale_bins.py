"""Shared scale-bin definitions for AS-NCCP R5 experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScaleBinSpec:
    """Immutable half-open scale bins with an inclusive final edge.

    ``edges=(0, 8, 12, 16, 20)`` yields four bins: [0, 8), [8, 12),
    [12, 16), and [16, 20]. Sizes are geometric-mean box sizes in pixels.
    """

    edges: tuple[float, ...] = (0.0, 8.0, 12.0, 16.0, 20.0)

    def __post_init__(self) -> None:
        edges = tuple(float(edge) for edge in self.edges)
        if len(edges) < 2:
            raise ValueError("scale bins require at least two edges")
        if not all(math.isfinite(edge) for edge in edges):
            raise ValueError("scale-bin edges must be finite")
        if edges[0] < 0.0:
            raise ValueError("the first scale-bin edge must be non-negative")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("scale-bin edges must be strictly increasing")
        object.__setattr__(self, "edges", edges)

    @property
    def num_bins(self) -> int:
        return len(self.edges) - 1

    @property
    def bin_ids(self) -> tuple[int, ...]:
        return tuple(range(self.num_bins))

    def index(self, size: float) -> int | None:
        """Return a bin id, or ``None`` when size is outside the configured range."""
        value = float(size)
        if not math.isfinite(value) or value < self.edges[0] or value > self.edges[-1]:
            return None
        for index, (lower, upper) in enumerate(zip(self.edges, self.edges[1:])):
            if value < upper or (index == self.num_bins - 1 and value <= upper):
                return index
        return None

    def contains(self, bin_id: int, size: float) -> bool:
        if bin_id < 0 or bin_id >= self.num_bins:
            return False
        return self.index(size) == bin_id

    def assign(self, sizes: Iterable[float]) -> list[int]:
        """Assign valid sizes and reject out-of-range values explicitly."""
        result = []
        for size in sizes:
            bin_id = self.index(size)
            if bin_id is None:
                raise ValueError(f"size {size!r} is outside scale-bin edges {self.edges}")
            result.append(bin_id)
        return result

    def to_dict(self) -> dict[str, object]:
        return {"edges": list(self.edges), "num_bins": self.num_bins}

    @classmethod
    def from_edges(cls, edges: Sequence[float]) -> "ScaleBinSpec":
        return cls(tuple(float(edge) for edge in edges))


DEFAULT_SCALE_BINS = ScaleBinSpec()

__all__ = ["DEFAULT_SCALE_BINS", "ScaleBinSpec"]
