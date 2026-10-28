"""Cross-view gradient inconsistency.                           Owner: Quỳnh Anh

Definition (from the Paper A overview, component v_i of s_geom):

    g_{i,c} = dL_c / d mu_i            (3D positional gradient from view c)
    v_i     = Tr(Cov_c[g_{i,c}]) / ( || E_c[g_{i,c}] ||^2 + delta )

over views c in V_i (views where Gaussian i contributes). Higher = views disagree more
= less reliable (hypothesis, to be tested — not assumed).

Uses population covariance and O(N) streaming moments. Counts < n_min receive zero
(no evidence of disagreement, NOT evidence of reliability). Counts are exported
separately. Measures amplitude variation as well as direction disagreement.
Fixed delta is the baseline; relative delta is an explicit optional ablation.
No training parameters, existing .grad buffers, or training accumulators are changed.
"""
from __future__ import annotations

import math

import torch

from ..losses import photometric_loss
from .common import SignalContext


def position_gradient_statistics(ctx: SignalContext, contrib_eps: float = 1e-3,
                                 lambda_dssim: float = 0.2) -> dict:
    """Float64 Welford moments, one render per view, constant memory in view count.

    Differentiate world means, including projected-center AND covariance paths.
    Unobserved views are excluded rather than treated as zero-gradient samples.
    """
    key = ("position_gradient_statistics", contrib_eps, lambda_dssim)
    if key in ctx.cache:
        return ctx.cache[key]
    n, dev = ctx.model.n, ctx.model.means.device
    count = torch.zeros(n, dtype=torch.int64, device=dev)
    mean = torch.zeros(n, 3, dtype=torch.float64, device=dev)
    m2 = torch.zeros(n, dtype=torch.float64, device=dev)
    for cam in ctx.cams:
        out = ctx.model.render(ctx.rasterize, cam, ctx.bg, return_contrib=True)
        loss = photometric_loss(out["image"], cam.image, lambda_dssim)
        gradient, = torch.autograd.grad(loss, ctx.model.means)
        with torch.no_grad():
            seen = out["contrib"] > contrib_eps
            g = gradient[seen].double()
            if not torch.isfinite(g).all():
                raise ValueError("non-finite world-position gradients")
            count[seen] += 1
            difference = g - mean[seen]
            updated = mean[seen] + difference / count[seen, None]
            m2[seen] += (difference * (g - updated)).sum(-1)
            mean[seen] = updated
    stats = {"count": count, "mean": mean,
             "variance_trace": (m2 / count.clamp_min(1)).clamp_min(0)}
    ctx.cache[key] = stats
    return stats


def compute(ctx: SignalContext, delta: float = 1e-12, n_min: int = 2,
            contrib_eps: float = 1e-3, lambda_dssim: float = 0.2,
            delta_rel: float = 0.0) -> torch.Tensor:
    """Effective delta = delta + delta_rel * median(||mean(g)||^2).

    The median uses only Gaussians observed in >= n_min views. Raw unnormalized
    scores are returned. No clipping is applied to large finite scores.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not math.isfinite(delta_rel) or delta_rel < 0:
        raise ValueError("delta_rel must be finite and nonnegative")
    if not isinstance(n_min, int) or n_min < 2:
        raise ValueError("n_min must be an integer >= 2")
    if not math.isfinite(contrib_eps) or contrib_eps < 0:
        raise ValueError("contrib_eps must be finite and nonnegative")
    if not 0 <= lambda_dssim <= 1:
        raise ValueError("lambda_dssim must be in [0, 1]")
    stats = position_gradient_statistics(ctx, contrib_eps, lambda_dssim)
    eligible = stats["count"] >= n_min
    mean_sq = stats["mean"].square().sum(-1)
    scale = mean_sq[eligible].median().item() if eligible.any() else 0.0
    effective_delta = delta + delta_rel * scale
    score = torch.zeros_like(mean_sq)
    score[eligible] = stats["variance_trace"][eligible] / (mean_sq[eligible] + effective_delta)
    score = score.float()
    if not torch.isfinite(score).all():
        raise ValueError("gradient inconsistency overflows float32; inspect gradients/delta")
    ctx.cache["gradient_inconsistency_stats"] = dict(stats, effective_delta=effective_delta)
    return score
