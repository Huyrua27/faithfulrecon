"""FisherRF-style information signal.                            Owner: Phát

Reference: Jiang et al., FisherRF (ECCV 2024). FisherRF uses the diagonal of the Fisher /
Hessian of the photometric loss over all radiance-field parameters (Laplace approximation).

Planned definition (verify against the paper and document differences):
    score_i = -log( Tr(F_i) + eps )     over all parameters of Gaussian i
    (little information = less reliable)

TODO(Phat):
  - [ ] which parameters FisherRF includes; per-Gaussian aggregation (trace? sum of diag?)
  - [ ] diagonal only is enough here: diag(F_i) = E[g_i ** 2] (cheaper than full blocks)
  - [ ] orientation check + test (contract: shape (N,), float32, finite)
"""
from __future__ import annotations

import torch

from .common import SignalContext


def compute(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-12) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat): FisherRF-style signal")
