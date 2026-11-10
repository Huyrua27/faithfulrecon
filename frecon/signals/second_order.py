"""Per-Gaussian Gauss–Newton / Fisher blocks via a Hutchinson-style estimator.   Owner: Phát

    F_i = sum_views sum_pixels J_{p,i}^T J_{p,i},      J_{p,i} = dI_p / d theta_i

Estimator: for Rademacher v (one +-1 per pixel and channel),
    g = J^T v = d<v, I>/d theta      (one backward pass)
    E[g_i g_i^T] = F_i               because E[v v^T] = I
Average g_i g_i^T over n_samples draws per view and sum over views.

TODO(Phat):
  - [ ] fisher_blocks(): params=("means",) -> (N,3,3); ("means","log_scales") -> (N,6,6).
        Note: scales are parameterized as log_scales in GaussianModel — document whether the
        block is w.r.t. log-scale or scale (PUP uses the activated scale).
  - [ ] cache in ctx.cache[("fisher", params, n_samples)] like common.view_statistics.
  - [ ] memory: accumulate in place; never keep per-pixel Jacobians.
  - [ ] test (tests/test_second_order.py): exact Jacobian on a tiny scene vs estimate,
        relative error decreasing roughly like 1/sqrt(n_samples).
"""
from __future__ import annotations

import torch

from .common import SignalContext


def fisher_blocks(ctx: SignalContext, params: tuple = ("means",), n_samples: int = 4) -> torch.Tensor:
    """(N, D, D) per-Gaussian Gauss–Newton blocks, D = sum of parameter sizes."""
    raise NotImplementedError("TODO(Phat): Hutchinson-style Fisher blocks")
