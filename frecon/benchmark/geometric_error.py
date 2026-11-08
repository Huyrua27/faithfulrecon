"""Continuous center-to-reference-sample geometric error."""
from __future__ import annotations

import torch

from .reference_geometry import ReferenceGeometry


@torch.no_grad()
def point_to_surface_distance(means: torch.Tensor, ref: ReferenceGeometry,
                             chunk_size: int = 512,
                             ref_chunk_size: int = 8192) -> torch.Tensor:
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if ref.points.ndim != 2 or ref.points.shape[1] != 3:
        raise ValueError("ref.points must have shape (M, 3)")
    if means.shape[0] == 0 or ref.points.shape[0] == 0:
        raise ValueError("means and reference points must be non-empty")
    if chunk_size <= 0 or ref_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if not torch.isfinite(means).all() or not torch.isfinite(ref.points).all():
        raise ValueError("Geometry must be finite")

    device = means.device
    query = means.to(dtype=torch.float32)
    reference = ref.points.to(device=device, dtype=torch.float32)
    result = torch.empty(query.shape[0], dtype=torch.float32, device=device)
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start:start + chunk_size]
        nearest = torch.full((chunk.shape[0],), float("inf"), device=device)
        for ref_start in range(0, reference.shape[0], ref_chunk_size):
            block = torch.cdist(chunk, reference[ref_start:ref_start + ref_chunk_size], p=2)
            nearest = torch.minimum(nearest, block.min(dim=1).values)
        result[start:start + chunk.shape[0]] = nearest
    return result
