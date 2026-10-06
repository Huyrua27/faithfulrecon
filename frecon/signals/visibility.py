"""Visibility signal.                                           Owner: Quỳnh Anh

score_i = -(number of training views in which Gaussian i contributes) + tiny random tie-break
(fewer observing views = less reliable)

Status: implemented and used in Phase 0 (not measurable in every seed, SNR_in 1.8 ± 0.6).
TODO(QuynhAnh):
  - [ ] "Contributes" = summed blending weight > 1e-3 in that view. Is this threshold sensible?
        Compare with a footprint-only definition (radius > 0) and with occlusion-aware counts.
  - [ ] Many Gaussians have count 0 (never visible): top-K then selects invisible Gaussians,
        so removal changes nothing. Decide whether to exclude them and document it.
  - [ ] Ties: integer counts -> tie-break is random; report how much of top-K is decided by ties.
  - [ ] Relation to the condition label m_i (Trường Thịnh): visibility is close to the
        definition of "under-observed" — avoid circular evaluation in D1.
"""
from __future__ import annotations

import torch

from .common import SignalContext, view_statistics


def compute(ctx: SignalContext) -> torch.Tensor:
    vis = view_statistics(ctx)["vis_count"]
    tie = 1e-3 * torch.rand(vis.numel(), generator=ctx.generator, device=vis.device)
    return (-vis + tie).float()
