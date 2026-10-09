"""Curvature term u_i and the candidate geometry reliability score s_geom.   Owner: Phát

From the Paper A overview — s_geom is a HYPOTHESIS to evaluate, not a proposed method:

    u_i = Tr( (F_i^mu + eps I)^{-1} )      3x3 position block (second_order.fisher_blocks)
    v_i = cross-view gradient inconsistency (Quỳnh Anh: gradient_inconsistency.py)
    s_geom(i) = exp(-lambda * u_i * v_i)    reliability; as a score (higher = less reliable)
                                            we use u_i * v_i directly — exp/lambda are monotone
                                            and do not change any rank-based metric.

Variants for ablation (each registered as its own signal name):
    curvature       u_i
    s_geom          u_i * v_i
    s_geom_ranksum  rank(u_i) + rank(v_i)

TODO(Phat):
  - [ ] curvature(): uses fisher_blocks(ctx, ("means",)); sensitivity to eps
  - [ ] s_geom(): needs v_i — import gradient_inconsistency.compute when Quỳnh Anh's version is
        merged; until then a temporary local version is allowed (do NOT edit her file)
  - [ ] rank-sum variant; position-only vs position+scale block ablation
"""
from __future__ import annotations

import torch

from .common import SignalContext


def curvature(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat): u_i")


def s_geom(ctx: SignalContext) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat): u_i * v_i")


def s_geom_ranksum(ctx: SignalContext) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat): rank(u_i) + rank(v_i)")
