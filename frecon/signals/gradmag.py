"""Gradient-magnitude signal.                                   Owner: Quỳnh Anh

score_i = mean over training views where Gaussian i is visible of
          || dL_c / d mu2d_i ||  (view-space positional gradient, NDC units)
This is the 3DGS densification statistic, recomputed once on the trained model.

Status: implemented and used in Phase 0 (measurable response, SNR_in ~4.7).
TODO(QuynhAnh):
  - [ ] Compare with the accumulator inside 3DGS training (frecon/train.py, train_step):
        same units? same visibility filter? 3DGS resets it after each densification.
  - [ ] Check normalization: does the score scale with Gaussian screen size / opacity?
        (confounder: large or opaque Gaussians get larger gradients)
  - [ ] Aggregation across views: mean vs max vs sum — document the choice and its effect.
  - [ ] Explain the Phase 0 result: measurable response but local geometry got WORSE than
        random removal (Gain_Dir local CD < 0). See docs/analysis/phase0_analysis.md.
"""
from __future__ import annotations

import torch

from .common import SignalContext, view_statistics


def compute(ctx: SignalContext) -> torch.Tensor:
    return view_statistics(ctx)["grad_mean"].float()
