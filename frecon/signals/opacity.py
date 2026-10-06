"""Opacity signal.                                              Owner: Quỳnh Anh

score_i = 1 - sigmoid(opacity_logit_i)        (higher = less reliable)

Status: implemented and used in Phase 0.
TODO(QuynhAnh):
  - [ ] Verify against the 3DGS definition of opacity (activation, where it is read).
  - [ ] Document confounders: opacity reset schedule, low-opacity Gaussians that never
        contribute (Phase 0: removal response at the noise floor, SNR_in ~1.1).
  - [ ] Decide whether to restrict to Gaussians that contribute in >= 1 view, and document it.
  - [ ] Unit test in tests/test_signals.py (shape, orientation, no NaN).
"""
from __future__ import annotations

import torch

from .common import SignalContext


def compute(ctx: SignalContext) -> torch.Tensor:
    return (1.0 - ctx.model.opacities).detach().float()
