"""Optional gsplat backend with the same interface as `torch_rast.rasterize`.

Notes:
- gsplat does not expose per-Gaussian blending weights, so `contrib` is approximated by
  the screen-space footprint area of visible Gaussians times opacity. Signals or masks
  that rely on `contrib` (visibility count) are therefore not identical across backends.
- The view-space gradient used by the densification signal is taken from
  `meta["means2d"]` (retain_grad), in pixel units, as in the torch backend.
"""
from __future__ import annotations

import torch

from ..camera import Camera


def rasterize(means, quats, scales, opacities, features, cam: Camera, bg: torch.Tensor,
              return_contrib: bool = False, **_):
    from gsplat import rasterization

    viewmat = torch.eye(4, device=means.device)
    viewmat[:3, :3] = cam.R
    viewmat[:3, 3] = cam.t
    K = torch.tensor([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]],
                     device=means.device, dtype=means.dtype)
    colors, alphas, meta = rasterization(
        means, quats / quats.norm(dim=-1, keepdim=True), scales, opacities, features,
        viewmat[None], K[None], cam.W, cam.H, packed=False,
        backgrounds=bg[None].to(features.dtype))
    means2d = meta["means2d"][0]
    radii = meta["radii"][0]
    if radii.dim() == 2:  # newer gsplat returns per-axis radii
        radii = radii.max(dim=-1).values
    radii = radii.float()
    out = {"image": colors[0], "alpha": alphas[0, ..., 0], "means2d": means2d,
           "radii": radii, "visible": radii > 0}
    if return_contrib:
        out["contrib"] = (radii ** 2 * opacities).detach() * (radii > 0)
    return out
