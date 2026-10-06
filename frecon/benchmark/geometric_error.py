"""Continuous per-Gaussian geometric error.                     Owner: Trường Thịnh

d_i = distance from Gaussian center mu_i to the reference surface.
A nearest-neighbor helper already exists: frecon.gaussians.nearest_dist(query, ref).

TODO(TruongThinh):
  - [ ] point_to_surface_distance(): center-to-nearest-reference-point (baseline). Make sure the
        reference is dense enough that sampling spacing << tau (report the spacing).
  - [ ] Optional: point-to-plane with normals; optional: account for Gaussian extent
        (a large Gaussian centered on the surface can still render off-surface).
  - [ ] Signed vs unsigned: in front of the surface (towards cameras) vs inside/behind.
        Floaters in sparse-view 3DGS are typically in front — keep the sign if cheap.
  - [ ] Unit test with a sphere: analytic distance vs computed distance.
"""
from __future__ import annotations

import torch

from .reference_geometry import ReferenceGeometry


def point_to_surface_distance(means: torch.Tensor, ref: ReferenceGeometry) -> torch.Tensor:
    """(N,) unsigned distance of each Gaussian center to the reference surface."""
    raise NotImplementedError("TODO(TruongThinh): geometric error d_i")
