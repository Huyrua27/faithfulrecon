"""Continuous per-Gaussian geometric error.
Distance is measured to the nearest sampled reference point,
not to an exact continuous mesh surface.
"""
from __future__ import annotations

import torch
from .reference_geometry import ReferenceGeometry


@torch.no_grad()
def point_to_surface_distance(
    means: torch.Tensor,
    ref: ReferenceGeometry,
    chunk_size: int = 512,
    ref_chunk_size: int = 8192,
) -> torch.Tensor:
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if ref.points.ndim != 2 or ref.points.shape[1] != 3:
        raise ValueError("ref.points must have shape (M, 3)")
    if means.shape[0] == 0 or ref.points.shape[0] == 0:
        raise ValueError("means and reference points must be non-empty")

    device = means.device
    ref_pts = ref.points.to(device=device, dtype=torch.float32)
    means = means.to(dtype=torch.float32)
    distances = torch.empty(means.shape[0], dtype=torch.float32, device=device)

    N = means.shape[0]
    M = ref_pts.shape[0]

    for i in range(0, N, chunk_size):
        chunk = means[i : i + chunk_size]
        best_dist = torch.full((chunk.shape[0],), float("inf"), device=device, dtype=torch.float32)
        for j in range(0, M, ref_chunk_size):
            ref_chunk = ref_pts[j : j + ref_chunk_size]
            block_dist = torch.cdist(chunk, ref_chunk, p=2)
            block_min = block_dist.min(dim=1).values
            best_dist = torch.minimum(best_dist, block_min)
        distances[i : i + chunk.shape[0]] = best_dist

    return distances
