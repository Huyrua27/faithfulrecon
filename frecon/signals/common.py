"""Shared pieces for reliability signals.

Interface contract (see docs/data_format.md):
  - every signal is a function `compute(ctx: SignalContext) -> torch.Tensor` of shape (N,),
    float32, one score per Gaussian, in checkpoint order (gaussian_id = row index);
  - orientation: HIGHER score = LESS reliable (selected first for removal);
  - no NaN/Inf in the output (use a documented fallback for Gaussians with no evidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..gaussians import GaussianModel
from ..losses import photometric_loss


@dataclass
class SignalContext:
    model: GaussianModel
    cams: list            # training cameras (with .image)
    rasterize: callable
    bg: torch.Tensor
    seed: int = 0
    cache: dict = field(default_factory=dict)  # shared per-view statistics, computed once

    @property
    def generator(self) -> torch.Generator:
        if "_gen" not in self.cache:
            self.cache["_gen"] = torch.Generator(device=self.model.means.device).manual_seed(self.seed)
        return self.cache["_gen"]


def view_statistics(ctx: SignalContext, contrib_eps: float = 1e-3, lambda_dssim: float = 0.2):
    """One pass over the training views, cached in ctx.cache["view_stats"]:
      grad_mean  (N,)  mean NDC view-space positional gradient norm over views where visible
      vis_count  (N,)  number of views where the Gaussian contributes (> contrib_eps)
    """
    if "view_stats" in ctx.cache:
        return ctx.cache["view_stats"]
    model = ctx.model
    n, dev = model.n, model.means.device
    grad_sum = torch.zeros(n, device=dev)
    vis_count = torch.zeros(n, device=dev)
    for cam in ctx.cams:
        for p in model.params.values():
            p.grad = None
        out = model.render(ctx.rasterize, cam, ctx.bg, return_contrib=True)
        out["means2d"].retain_grad()
        photometric_loss(out["image"], cam.image, lambda_dssim).backward()
        with torch.no_grad():
            g = out["means2d"].grad
            g_ndc = torch.stack([g[:, 0] * cam.W * 0.5, g[:, 1] * cam.H * 0.5], -1).norm(dim=-1)
            seen = out["contrib"] > contrib_eps
            grad_sum[seen] += g_ndc[seen]
            vis_count[seen] += 1
    for p in model.params.values():
        p.grad = None
    stats = {"grad_mean": grad_sum / vis_count.clamp_min(1), "vis_count": vis_count}
    ctx.cache["view_stats"] = stats
    return stats


def top_fraction(score: torch.Tensor, frac: float) -> torch.Tensor:
    """Boolean mask of the top `frac` of Gaussians by score (the least reliable ones)."""
    k = max(1, int(round(frac * score.numel())))
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask[torch.topk(score, k).indices] = True
    return mask
