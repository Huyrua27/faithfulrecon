"""FisherRF-style information signal.                            Owner: Phát

Reference: Jiang et al., FisherRF (ECCV 2024), https://arxiv.org/abs/2311.17874.
This is an adapted per-Gaussian information baseline, not the original EIG/uncertainty map.
The authors' default uncertainty renderer selects color/SH fields; here we use all 14 STORED
parameters of this SH-degree-0 model (position, log-scale, quaternion, opacity/color logits).

Definition:
    score_i = -log( Tr(F_i) + eps )     over all parameters of Gaussian i
    (little information = less reliable)

The trace mixes parameter units and is not invariant to reparameterization. Signals use
training cameras only. The diagonal estimator avoids full 14x14 blocks.
"""
from __future__ import annotations

import torch

from .common import SignalContext
from .second_order import finite_score, fisher_diagonal, positive_scalar


def compute(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-12) -> torch.Tensor:
    positive_scalar(eps, "eps")
    information = fisher_diagonal(ctx, tuple(ctx.model.params), n_samples)
    return finite_score(-(information.double().sum(-1) + eps).log())
