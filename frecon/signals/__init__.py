"""Per-Gaussian reliability signals. Convention: higher score = LESS reliable.

Add a signal: create `<name>.py` with `compute(ctx) -> Tensor[N]`, register it in REGISTRY,
add a test in tests/test_signals.py. See docs/data_format.md for the output contract.
"""
from __future__ import annotations

import torch

from . import gradient_inconsistency, gradmag, opacity, visibility
from .common import SignalContext, top_fraction, view_statistics


def _random(ctx: SignalContext) -> torch.Tensor:
    return torch.rand(ctx.model.n, generator=ctx.generator, device=ctx.model.means.device)


REGISTRY = {
    "random": _random,
    "opacity": opacity.compute,
    "gradmag": gradmag.compute,
    "visibility": visibility.compute,
    "gradient_inconsistency": gradient_inconsistency.compute,
}

SIGNALS = ("random", "opacity", "gradmag", "visibility")  # Phase 0 set


def compute_signals(model, cams, rasterize, bg, names=SIGNALS, seed: int = 0) -> dict:
    """Compute several signals sharing one SignalContext (per-view passes are cached)."""
    ctx = SignalContext(model=model, cams=cams, rasterize=rasterize, bg=bg, seed=seed)
    out = {}
    for name in names:
        if name not in REGISTRY:
            raise ValueError(f"unknown signal: {name} (known: {sorted(REGISTRY)})")
        score = REGISTRY[name](ctx)
        if score.shape != (model.n,) or not torch.isfinite(score).all():
            raise ValueError(f"signal '{name}' violates the output contract (shape (N,), finite)")
        out[name] = score
    if "view_stats" in ctx.cache:
        out["_vis_count"] = ctx.cache["view_stats"]["vis_count"]
    return out


__all__ = ["REGISTRY", "SIGNALS", "SignalContext", "compute_signals", "top_fraction", "view_statistics"]
