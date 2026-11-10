"""Gradient-magnitude signal.                                   Owner: Quỳnh Anh

score_i = mean over training views where Gaussian i is visible of
          || dL_c / d mu2d_i ||  (view-space positional gradient, NDC units)
Related to the 3DGS densification statistic, recomputed once on a fixed checkpoint.

Training uses footprint visibility and a history of changing model parameters; this
signal uses contribution > contrib_eps on the frozen model. Same NDC conversion,
different sampling/filter. Unobserved Gaussians receive zero; inspect counts too.
Mean of norms is retained as baseline; opacity/footprint/loss are confounders.
Phase 0 response is measurable, but local geometry becomes worse than random removal.
See docs/research/QuynhAnh_signal_research.md for interpretation and ablations.
"""
from __future__ import annotations

import torch

from .common import SignalContext, view_statistics


def compute(ctx: SignalContext, contrib_eps: float = 1e-3,
            lambda_dssim: float = 0.2) -> torch.Tensor:
    return view_statistics(ctx, contrib_eps, lambda_dssim)["grad_mean"].float()
