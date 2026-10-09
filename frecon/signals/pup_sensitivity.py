"""PUP 3D-GS-style sensitivity signal.                           Owner: Phát

Reference: Hanson et al., PUP 3D-GS (CVPR 2025). PUP scores each Gaussian by the log-determinant
of its Fisher block over spatial parameters (mean and scale) and prunes low-sensitivity Gaussians.

Planned definition (verify against the paper and document differences):
    F_i  = 6x6 block over (mu_i, scale_i)      (second_order.fisher_blocks)
    score_i = -log det(F_i + eps * I)          (low sensitivity = pruned first = less reliable)

TODO(Phat):
  - [ ] exact PUP parameterization (activated scale vs log-scale; any normalization)
  - [ ] numerical stability of log det for near-singular blocks (eps, slogdet)
  - [ ] note: the estimator here replaces PUP's CUDA kernel — report agreement on a tiny scene
  - [ ] test (contract + orientation)
"""
from __future__ import annotations

import torch

from .common import SignalContext


def compute(ctx: SignalContext, n_samples: int = 4, eps: float = 1e-8) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat): PUP-style sensitivity")
