"""Visibility signal.                                           Owner: Quỳnh Anh

score_i = -(number of training views in which Gaussian i contributes) + tiny random tie-break
(fewer observing views = less reliable)

Contributes = summed blending weight > 1e-3 (torch backend, occlusion-aware).
The gsplat proxy is different. Zero-count Gaussians stay in the baseline pool.
Legacy tie jitter uses the shared context generator, so score order affects ties;
seed and signal order must be recorded. Opacity, footprint, occlusion and view layout
are confounders; this is trained-model support, not reference-surface coverage m_i.
See docs/research/QuynhAnh_signal_research.md for interpretation and ablations.
"""
from __future__ import annotations

import torch

from .common import SignalContext, view_statistics


def compute(ctx: SignalContext, contrib_eps: float = 1e-3,
            lambda_dssim: float = 0.2) -> torch.Tensor:
    vis = view_statistics(ctx, contrib_eps, lambda_dssim)["vis_count"]
    tie = 1e-3 * torch.rand(vis.numel(), generator=ctx.generator, device=vis.device)
    return (-vis + tie).float()
