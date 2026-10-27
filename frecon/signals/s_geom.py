"""Curvature term u_i and the candidate geometry reliability score s_geom.   Owner: Phát

From the Paper A overview — s_geom is a HYPOTHESIS to evaluate, not a proposed method:

    u_i = Tr( (F_i^mu + eps I)^{-1} )      3x3 position block (second_order.fisher_blocks)
    v_i = cross-view gradient inconsistency (Quỳnh Anh: gradient_inconsistency.py)
    s_geom(i) = exp(-lambda * u_i * v_i)    reliability; as a score (higher = less reliable)
                                            we use u_i * v_i directly — exp/lambda are monotone
                                            and preserve per-Gaussian ranks. This does not
                                            imply invariance after blending pixel score maps.

Variants for ablation (each registered as its own signal name):
    curvature       u_i
    s_geom          u_i * v_i
    s_geom_ranksum  rank(u_i) + rank(v_i)

The local inconsistency fallback is used only while Quỳnh Anh's function is unimplemented.
Its provider and unknown-view convention are exported in metadata. Scale-block variants and
failure/predictive validity are later ablations, not established by these definitions.
"""
from __future__ import annotations

from numbers import Integral

import torch

from ..losses import photometric_loss
from . import gradient_inconsistency
from .common import SignalContext
from .second_order import damped_eigenvalues, finite_score, fisher_blocks, positive_scalar


def curvature(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8) -> torch.Tensor:
    return finite_score(damped_eigenvalues(fisher_blocks(ctx, ("means",), n_samples), eps).reciprocal().sum(-1))


def s_geom(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8,
           delta: float = 1e-12, n_min: int = 2) -> torch.Tensor:
    return finite_score(curvature(ctx, n_samples, eps).double() * inconsistency(ctx, delta, n_min).double())


def s_geom_ranksum(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8,
                   delta: float = 1e-12, n_min: int = 2) -> torch.Tensor:
    return finite_score(average_rank(curvature(ctx, n_samples, eps)) +
                        average_rank(inconsistency(ctx, delta, n_min)))


def average_rank(score):
    """Average ranks for ties in [0,1]; constant arrays receive rank 0.5."""
    if score.numel() <= 1:
        return torch.full_like(score, 0.5)
    _, inverse, counts = torch.unique(score, sorted=True, return_inverse=True, return_counts=True)
    before = counts.cumsum(0) - counts
    ranks = before.float() + (counts.float() - 1) * 0.5
    return ranks[inverse] / (score.numel() - 1)


@torch.enable_grad()
def inconsistency(ctx: SignalContext, delta: float = 1e-12, n_min: int = 2,
                  contrib_eps: float = 1e-3, lambda_dssim: float = 0.2) -> torch.Tensor:
    """Use Quỳnh Anh's implementation when available; otherwise use this local fallback.

    Population gradient variance trace / (squared mean + delta), over contributing training
    views. Fewer than n_min views receive v=1 (unknown), not v=0 (perfect agreement). Provider
    and unsupported count are recorded in ctx.cache. The fallback preserves model .grad.
    """
    positive_scalar(delta, "delta")
    if isinstance(n_min, bool) or not isinstance(n_min, Integral) or n_min < 2:
        raise ValueError("n_min must be an integer >= 2")
    if not 0 <= contrib_eps < float("inf") or not 0 <= lambda_dssim <= 1:
        raise ValueError("invalid contribution threshold or lambda_dssim")
    key = ("phat_inconsistency", delta, n_min, contrib_eps, lambda_dssim)
    if key in ctx.cache:
        return ctx.cache[key]
    if contrib_eps == 1e-3 and lambda_dssim == 0.2:
        try:
            score = finite_score(gradient_inconsistency.compute(ctx, delta=delta, n_min=n_min))
            ctx.cache[key] = score
            ctx.cache["phat_inconsistency_meta"] = {"provider": "quynhanh"}
            return score
        except NotImplementedError:
            pass
    if not ctx.cams or any(cam.image is None for cam in ctx.cams):
        raise ValueError("gradient inconsistency requires training cameras with GT images")
    n, device = ctx.model.n, ctx.model.means.device
    total = torch.zeros(n, 3, device=device, dtype=torch.float64)
    squared = torch.zeros(n, device=device, dtype=torch.float64)
    count = torch.zeros(n, device=device, dtype=torch.long)
    for cam in ctx.cams:
        out = ctx.model.render(ctx.rasterize, cam, ctx.bg, return_contrib=True)
        loss = photometric_loss(out["image"], cam.image, lambda_dssim)
        grad = torch.autograd.grad(loss, ctx.model.means, allow_unused=True)[0]
        if grad is None:
            continue
        grad = grad.detach().double()
        seen = out["contrib"].detach() > contrib_eps
        total[seen] += grad[seen]
        squared[seen] += grad[seen].square().sum(-1)
        count[seen] += 1
    mean_norm2 = (total / count.clamp_min(1)[:, None]).square().sum(-1)
    variance = (squared / count.clamp_min(1) - mean_norm2).clamp_min(0)
    score = finite_score(torch.where(count >= n_min, variance / (mean_norm2 + delta), 1.0))
    ctx.cache[key] = score
    ctx.cache["phat_inconsistency_meta"] = {
        "provider": "temporary_phat", "covariance": "population", "fallback": 1.0,
        "unsupported_count": int((count < n_min).sum()), "n_min": n_min,
        "delta": delta, "contrib_eps": contrib_eps, "lambda_dssim": lambda_dssim,
    }
    return score
