"""Pure-PyTorch Gaussian splatting rasterizer.

Forward model of 3DGS (Kerbl et al. 2023): EWA projection with a 0.3 px low-pass,
3-sigma screen footprint, per-pixel depth ordering, front-to-back alpha compositing with
alpha clamped to 0.99 and contributions below 1/255 discarded.

Instead of evaluating every Gaussian of a 16x16 tile at every pixel of the tile (what the
CUDA kernel does), we enumerate only the (Gaussian, pixel) pairs inside each footprint
whose alpha is >= 1/255, sort them by (pixel, depth), and composite with a segmented
log-transmittance cumsum. Everything is differentiated by autograd. Differences from the
CUDA rasterizer: no early ray termination at T < 1e-4 (we keep all contributions), and
depth ordering is per pixel rather than per tile. Both are small for opaque scenes.
"""
from __future__ import annotations

import torch

from ..camera import Camera

NEAR = 0.01
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def project(means, quats, scales, cam: Camera):
    """EWA projection. Returns screen means (N,2), conics (N,3), radii (N,), depths (N,), valid (N,)."""
    R, t = cam.R, cam.t
    p = means @ R.T + t
    x, y, z = p.unbind(-1)
    zc = z.clamp_min(NEAR)

    Rq = quat_to_rotmat(quats)
    M = Rq * scales[:, None, :]
    # elementwise products: batched 3x3 matmuls (bmm) are slow on small GPUs
    cov3 = (M[:, :, None, :] * M[:, None, :, :]).sum(-1)

    # Clamp the Jacobian evaluation point to 1.3x the frustum, as in 3DGS.
    lim_x = 1.3 * (0.5 * cam.W / cam.fx)
    lim_y = 1.3 * (0.5 * cam.H / cam.fy)
    tx = (x / zc).clamp(-lim_x, lim_x) * zc
    ty = (y / zc).clamp(-lim_y, lim_y) * zc
    zeros = torch.zeros_like(zc)
    J = torch.stack([
        cam.fx / zc, zeros, -cam.fx * tx / zc ** 2,
        zeros, cam.fy / zc, -cam.fy * ty / zc ** 2,
    ], dim=-1).reshape(-1, 2, 3)
    T = (J.reshape(-1, 3) @ R).reshape(-1, 2, 3)
    TC = (T[:, :, :, None] * cov3[:, None, :, :]).sum(2)
    cov2 = (TC[:, :, None, :] * T[:, None, :, :]).sum(-1)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)

    u = cam.fx * x / zc + cam.cx
    v = cam.fy * y / zc + cam.cy
    means2d = torch.stack([u, v], dim=-1)

    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
        radii = torch.ceil(3.0 * torch.sqrt(lam))
        valid = (z > NEAR) & (u + radii >= 0) & (u - radii < cam.W) & (v + radii >= 0) & (v - radii < cam.H)
        radii = torch.where(valid, radii, torch.zeros_like(radii))
    return means2d, conic, radii, z, valid


def _alpha(dx, dy, con, opa):
    power = -0.5 * (con[:, 0] * dx * dx + con[:, 2] * dy * dy) - con[:, 1] * dx * dy
    alpha = (opa * torch.exp(power.clamp_max(0.0))).clamp_max(ALPHA_MAX)
    return alpha, power


@torch.no_grad()
def _pairs(means2d, conic, opac, radii, valid, W, H, max_pairs_per_chunk=1 << 24):
    """Enumerate (Gaussian, pixel) pairs with alpha >= 1/255 inside each 3-sigma box."""
    dev = means2d.device
    gid = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if gid.numel() == 0:
        e = torch.zeros(0, dtype=torch.long, device=dev)
        return e, e
    u, v, r = means2d[gid, 0], means2d[gid, 1], radii[gid]
    # pixel centers are at integer + 0.5
    x0 = torch.ceil(u - r - 0.5).clamp(0, W - 1).long()
    x1 = torch.floor(u + r - 0.5).clamp(0, W - 1).long()
    y0 = torch.ceil(v - r - 0.5).clamp(0, H - 1).long()
    y1 = torch.floor(v + r - 0.5).clamp(0, H - 1).long()
    nx = (x1 - x0 + 1).clamp_min(0)
    ny = (y1 - y0 + 1).clamp_min(0)
    cnt = nx * ny
    # process Gaussians in chunks so that the candidate list stays bounded
    csum = torch.cumsum(cnt, 0)
    out_g, out_p = [], []
    start = 0
    n = gid.numel()
    while start < n:
        base = int(csum[start - 1]) if start > 0 else 0
        end = int(torch.searchsorted(csum, torch.tensor(base + max_pairs_per_chunk, device=dev), right=True))
        end = max(end, start + 1)
        sl = slice(start, end)
        c = cnt[sl]
        total = int(c.sum())
        if total > 0:
            rep = torch.repeat_interleave(torch.arange(start, end, device=dev), c)
            offs = torch.cumsum(c, 0) - c
            k = torch.arange(total, device=dev) - offs[rep - start]
            px = x0[rep] + k % nx[rep]
            py = y0[rep] + torch.div(k, nx[rep], rounding_mode="floor")
            g = gid[rep]
            alpha, power = _alpha(px.float() + 0.5 - means2d[g, 0], py.float() + 0.5 - means2d[g, 1],
                                  conic[g], opac[g])
            keep = (alpha >= ALPHA_MIN) & (power <= 0)
            out_g.append(g[keep])
            out_p.append((py * W + px)[keep])
        start = end
    return torch.cat(out_g), torch.cat(out_p)


def rasterize(means, quats, scales, opacities, features, cam: Camera, bg: torch.Tensor,
              return_contrib: bool = False, **_):
    """Render `features` (N,C). Returns dict with image (H,W,C), alpha (H,W), means2d (N,2),
    radii (N,), visible (N,), and optionally contrib (N,): summed blending weight per Gaussian."""
    N, C = features.shape
    dev = means.device
    W, H = cam.W, cam.H
    P = W * H
    means2d, conic, radii, depths, valid = project(means, quats, scales, cam)

    with torch.no_grad():
        pg, pp = _pairs(means2d.detach(), conic.detach(), opacities.detach(), radii, valid, W, H)
        rank = torch.empty(N, dtype=torch.long, device=dev)
        rank.scatter_(0, torch.argsort(depths), torch.arange(N, device=dev))
        order = torch.argsort(pp * N + rank[pg])
        pg, pp = pg[order], pp[order]
        # index of the first pair of each pair's pixel segment
        per_pix = torch.bincount(pp, minlength=P)
        seg_start = (torch.cumsum(per_pix, 0) - per_pix)[pp]

    packed = torch.cat([means2d, conic, opacities[:, None], features], dim=-1)
    g = torch.index_select(packed, 0, pg)
    px = (pp % W).float() + 0.5
    py = torch.div(pp, W, rounding_mode="floor").float() + 0.5
    alpha, _ = _alpha(px - g[:, 0], py - g[:, 1], g[:, 2:5], g[:, 5])

    # segmented exclusive cumsum of log(1 - alpha) along each pixel's depth-sorted list
    # (float64: the global cumsum reaches large magnitudes before the per-segment subtraction)
    logt = torch.log1p(-alpha)
    ld = logt.double()
    cs = torch.cumsum(ld, 0)  # inclusive
    # sum of logt over the pairs in front of this one within the same pixel
    excl = (cs - ld) - torch.index_select(cs - ld, 0, seg_start)
    w = alpha * torch.exp(excl.float())

    color = torch.zeros(P, C, device=dev, dtype=features.dtype).index_add(0, pp, w[:, None] * g[:, 6:])
    log_tfin = torch.zeros(P, device=dev, dtype=logt.dtype).index_add(0, pp, logt)
    t_fin = torch.exp(log_tfin)
    img = color + t_fin[:, None] * bg.to(features.dtype)

    out = {"image": img.reshape(H, W, C), "alpha": (1 - t_fin).reshape(H, W),
           "means2d": means2d, "radii": radii, "visible": radii > 0}
    if return_contrib:
        with torch.no_grad():
            out["contrib"] = torch.zeros(N, device=dev).index_add(0, pg, w.detach())
    return out
