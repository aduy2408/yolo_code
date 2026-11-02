"""Project-owned gradient aggregation primitives for tiny-object interventions.

The functions in this module are intentionally independent of Ultralytics. They
operate on already-computed per-object gradients so they can be unit-tested
without constructing a detector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mode_balanced_gradient(
    gradients: torch.Tensor,
    *,
    n_modes: int = 2,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate gradients with equal weight per discovered direction mode.

    Args:
        gradients: Tensor shaped ``[N, ...]`` containing raw per-object grads.
        n_modes: Maximum number of cosine-direction modes.
        iterations: Number of small, deterministic cosine-k-means iterations.

    Returns:
        A tuple ``(aggregate, assignments)``. The aggregate has the same shape
        as one gradient and assignments has shape ``[N]``.

    This is a torch-only implementation rather than a sklearn dependency. It
    uses farthest-point cosine initialization, then assigns each object to its
    nearest normalized centroid. Empty clusters are dropped from the final
    equal-weight average.
    """
    if gradients.ndim < 2:
        raise ValueError("gradients must have shape [N, ...]")
    if gradients.shape[0] == 0:
        raise ValueError("gradients must contain at least one object")
    if n_modes < 1:
        raise ValueError("n_modes must be positive")

    raw = gradients.reshape(gradients.shape[0], -1)
    directions = F.normalize(raw, dim=1, eps=1e-12)
    modes = min(int(n_modes), raw.shape[0])
    if modes == 1:
        return gradients.mean(dim=0), torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device)

    # Farthest-point initialization avoids a random seed in the training loop.
    centers = [directions[0]]
    min_distance = 1.0 - directions @ centers[0]
    for _ in range(1, modes):
        index = int(min_distance.argmax().item())
        centers.append(directions[index])
        min_distance = torch.minimum(min_distance, 1.0 - directions @ centers[-1])
    centroids = torch.stack(centers)

    assignments = torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device)
    for _ in range(max(int(iterations), 1)):
        assignments = (directions @ centroids.T).argmax(dim=1)
        updated = []
        for mode in range(modes):
            members = directions[assignments == mode]
            updated.append(
                F.normalize(members.mean(dim=0), dim=0, eps=1e-12)
                if members.numel()
                else centroids[mode]
            )
        new_centroids = torch.stack(updated)
        if torch.equal(assignments, (directions @ new_centroids.T).argmax(dim=1)):
            centroids = new_centroids
            break
        centroids = new_centroids

    mode_means = [
        raw[assignments == mode].mean(dim=0)
        for mode in range(modes)
        if (assignments == mode).any()
    ]
    aggregate = torch.stack(mode_means).mean(dim=0).reshape_as(gradients[0])
    return aggregate, assignments
