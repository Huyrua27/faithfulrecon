"""Opacity signal.                                              Owner: Quỳnh Anh

score_i = 1 - sigmoid(opacity_logit_i)        (higher = less reliable)

Activated opacity is read from GaussianModel; not alpha after pixel-footprint blending.
All checkpoint Gaussians are included. Low opacity often means low rendering influence,
not geometric error. Reset/pruning schedules and the opacity gate in floater labels
are confounders. Active-only selection is a proposed ablation, not the baseline.
See docs/research/QuynhAnh_signal_research.md and tests/test_signals.py.
"""
from __future__ import annotations

import torch

from .common import SignalContext


def compute(ctx: SignalContext) -> torch.Tensor:
    return (1.0 - ctx.model.opacities).detach().float()
