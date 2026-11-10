"""Phát's second-order scores and scores.npz handoff; leaves the shared exporter untouched."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import fisher, pup_sensitivity, s_geom
from .common import SignalContext
from .second_order import finite_score

NAMES = ("fisher", "pup", "curvature", "gradient_inconsistency", "s_geom", "s_geom_ranksum")


def compute(model, cams, rasterize, bg, names=NAMES, seed=0, n_samples=16, eps=1e-8,
            fisher_eps=1e-12, delta=1e-12, n_min=2, scale_basis="scale"):
    """Return scores and per-signal options/provider metadata using one shared context."""
    names = tuple(names)
    if not names or len(set(names)) != len(names) or set(names) - set(NAMES):
        raise ValueError(f"signals must be unique names from {NAMES}")
    ctx = SignalContext(model, cams, rasterize, bg, seed)
    scores, options = {}, {}
    for name in names:
        if name == "fisher":
            value = fisher.compute(ctx, n_samples, fisher_eps)
            settings = dict(n_samples=n_samples, eps=fisher_eps, params=list(model.params), basis="stored")
        elif name == "pup":
            value = pup_sensitivity.compute(ctx, n_samples, eps, scale_basis)
            settings = dict(n_samples=n_samples, eps=eps, scale_basis=scale_basis)
        elif name == "curvature":
            value = s_geom.curvature(ctx, n_samples, eps)
            settings = dict(n_samples=n_samples, eps=eps, basis="position")
        elif name == "gradient_inconsistency":
            value = s_geom.inconsistency(ctx, delta, n_min)
            settings = dict(delta=delta, n_min=n_min)
        else:
            function = s_geom.s_geom if name == "s_geom" else s_geom.s_geom_ranksum
            value = function(ctx, n_samples, eps, delta, n_min)
            settings = dict(n_samples=n_samples, eps=eps, delta=delta, n_min=n_min)
        scores[name] = finite_score(value)
        if scores[name].shape != (model.n,):
            raise ValueError(f"{name} violates the (N,) score contract")
        options[name] = dict(settings, protocol="phat_second_order_v1", higher="less_reliable",
                             renderer=rasterize.__module__)
    for name in ("gradient_inconsistency", "s_geom", "s_geom_ranksum"):
        if name in options:
            options[name]["inconsistency"] = ctx.cache["phat_inconsistency_meta"]
    return scores, options


def save_scores(path, scores, meta, overwrite=False):
    """Atomically merge into scores.npz, validating checkpoint identity and preserving baselines.

    Replacing any existing signal requires overwrite=True. Never accept a legacy/mismatched
    file without its identity metadata; mixing checkpoints can otherwise yield plausible metrics.
    """
    path = Path(path)
    identity = ("scene", "condition", "seed", "checkpoint_sha256", "n_gaussians")
    if any(key not in meta for key in identity) or "signal_params" not in meta:
        raise ValueError("missing score identity/options metadata")
    n = meta["n_gaussians"]
    arrays = {name: (value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))
              for name, value in scores.items()}
    arrays = {name: value.astype(np.float32) for name, value in arrays.items()}
    existing, combined_meta = {}, dict(meta)
    if path.exists():
        with np.load(path, allow_pickle=True) as archive:
            old_meta = archive["meta"].item()
            if any(old_meta.get(key) != meta[key] for key in identity):
                raise ValueError("existing scores belong to a different checkpoint/run")
            if (old_meta.get("scene_config_sha256") is not None and
                    old_meta["scene_config_sha256"] != meta.get("scene_config_sha256")):
                raise ValueError("existing scores use a different scene/camera configuration")
            existing = {name: archive[name].copy() for name in archive.files if name != "meta"}
        overlap = set(existing) & set(arrays)
        if overlap and not overwrite:
            raise FileExistsError(f"signals already exist: {sorted(overlap)}; use --overwrite to replace")
        combined_meta = dict(old_meta, **meta)
        combined_meta["signal_params"] = dict(old_meta.get("signal_params", {}), **meta["signal_params"])
    merged = dict(existing, **arrays)
    if any(value.shape != (n,) or value.dtype != np.float32 or not np.isfinite(value).all()
           for value in merged.values()):
        raise ValueError("scores must be finite float32 (N,) arrays in checkpoint order")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **merged, meta=np.array(combined_meta, dtype=object))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
