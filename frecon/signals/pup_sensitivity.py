"""PUP 3D-GS-style sensitivity signal.                           Owner: Phát

Reference: Hanson et al., PUP 3D-GS (CVPR 2025), https://arxiv.org/abs/2406.10219.
PUP scores each Gaussian by the log-determinant
of its Fisher block over spatial parameters (mean and scale) and prunes low-sensitivity Gaussians.

Definition (stochastic adaptation, not a reproduction of PUP's CUDA kernel):
    F_i  = 6x6 block over (mu_i, scale_i)      (second_order.fisher_blocks)
    score_i = -log det(F_i + eps * I)          (low sensitivity = pruned first = less reliable)

Default: activated scale, via chain-rule conversion of the native log-scale block.
scale_basis="log_scale" is a different ablation. Absolute damping is applied after conversion.
A float64 symmetric eigenspectrum gives a stable log determinant; negative roundoff is
clipped to zero before damping. No patch downsampling is used.
"""
from __future__ import annotations

import torch

from .common import SignalContext
from .second_order import damped_eigenvalues, finite_score, fisher_blocks


def compute(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8,
            scale_basis: str = "scale") -> torch.Tensor:
    if scale_basis not in ("scale", "log_scale"):
        raise ValueError("scale_basis must be 'scale' or 'log_scale'")
    blocks = fisher_blocks(ctx, ("means", "log_scales"), n_samples).double()
    if scale_basis == "scale":
        scales = ctx.model.params["log_scales"].detach().double().exp()
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("activated scales must be finite and positive")
        factors = torch.cat([torch.ones_like(scales), scales.reciprocal()], -1)
        blocks = blocks * factors[:, :, None] * factors[:, None, :]
    return finite_score(-damped_eigenvalues(blocks, eps).log().sum(-1))
