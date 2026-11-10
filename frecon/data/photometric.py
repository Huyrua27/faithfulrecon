"""Per-view photometric perturbations for controlled lighting-inconsistency experiments.

Simulates a camera pipeline that differs between training images (auto exposure, white balance,
tone curve). Inside the object mask, image k becomes

    I'_k = clip(g_k * w_k * I_k, 0, 1) ** gamma_k

    g_k     = 2 ** e_k,           e_k ~ U(-exposure_stops, exposure_stops)
    w_k     = exp(u_k) per RGB,   u_k ~ U(-wb, wb)^3
    gamma_k = exp(v_k),           v_k ~ U(-gamma, gamma)

Pixels outside the mask (the composited background) are left unchanged, so the perturbation only
affects the object. Parameters depend only on `seed` and the view's position in the list, so every
training seed sees the same perturbed images. Config (under `scene:`):

    photometric: {exposure_stops: 0.5, wb: 0.05, gamma: 0.1, seed: 0}
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch


def sample_params(n: int, exposure_stops: float = 0.0, wb: float = 0.0, gamma: float = 0.0,
                  seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    e = rng.uniform(-exposure_stops, exposure_stops, n)
    u = rng.uniform(-wb, wb, (n, 3))
    v = rng.uniform(-gamma, gamma, n)
    return [{"gain": float(2.0 ** e[k]), "wb": [float(x) for x in np.exp(u[k])], "gamma": float(np.exp(v[k]))}
            for k in range(n)]


def apply(image: torch.Tensor, mask: torch.Tensor, p: dict) -> torch.Tensor:
    """image (H, W, 3) in [0, 1]; mask (H, W) bool; p from sample_params."""
    scale = p["gain"] * torch.tensor(p["wb"], dtype=image.dtype, device=image.device)
    out = (image * scale).clamp(0, 1) ** p["gamma"]
    return torch.where(mask[..., None], out, image)


def perturb_views(cams, masks, cfg: dict):
    """Return (perturbed cams, params). `masks` are per-camera (H, W) bool arrays or tensors."""
    params = sample_params(len(cams), cfg.get("exposure_stops", 0.0), cfg.get("wb", 0.0),
                           cfg.get("gamma", 0.0), cfg.get("seed", 0))
    out = []
    for cam, m, p in zip(cams, masks, params):
        m = torch.as_tensor(np.asarray(m), device=cam.image.device).bool()
        out.append(replace(cam, image=apply(cam.image, m, p)))
    return out, params
