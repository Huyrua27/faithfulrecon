"""Cross-view gradient inconsistency.                           Owner: Quỳnh Anh

Definition (from the Paper A overview, component v_i of s_geom):

    g_{i,c} = dL_c / d mu_i            (3D positional gradient from view c)
    v_i     = Tr(Cov_c[g_{i,c}]) / ( || E_c[g_{i,c}] ||^2 + delta )

over views c in V_i (views where Gaussian i contributes). Higher = views disagree more
= less reliable (hypothesis, to be tested — not assumed).

Status: NOT implemented.
TODO(QuynhAnh):
  - [ ] Per-view gradients of means (3D) for every Gaussian: one backward per view, like
        common.view_statistics. Accumulate sum g, sum g g^T (or sum ||g||^2 for the trace)
        and counts — avoid storing all per-view gradients (memory).
  - [ ] Gaussians with |V_i| < n_min views: decide fallback score (document; no NaN).
  - [ ] delta: choose a scale-aware value (e.g. relative to median ||E g||^2); ablate.
  - [ ] Unit test: synthetic case where two views give opposite gradients -> high score.
  - [ ] Register "gradient_inconsistency" in configs (intervention.signals) once it passes tests.
"""
from __future__ import annotations

import torch

from .common import SignalContext


def compute(ctx: SignalContext, delta: float = 1e-12, n_min: int = 2) -> torch.Tensor:
    raise NotImplementedError("TODO(QuynhAnh): cross-view gradient inconsistency")
