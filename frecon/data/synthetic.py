"""Procedural synthetic scene with exact reference geometry.

The ground-truth scene is a dense layer of flat, nearly opaque Gaussians sampled on the
surfaces of a few textured primitives (sphere, box, thin rods). GT images are rendered
from it with the same rasterizer. Because the geometry is analytic, the reference
surface is known exactly, which gives the benchmark-defined labels (condition m_i,
outcome f_i) needed by the protocol, and any camera configuration can be generated.

Limitation: GT images come from a Gaussian renderer, so the image formation model is
in-distribution for 3DGS. Use `blender.py` (NeRF-Synthetic) for mesh-rendered data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..camera import Camera, orbit_cameras
from ..gaussians import GaussianModel, inverse_sigmoid


@dataclass
class SyntheticSceneConfig:
    resolution: int = 160
    fov_deg: float = 40.0
    cam_radius: float = 4.0
    n_gt_gaussians: int = 40_000
    n_ref_points: int = 60_000
    texture_freq: float = 10.0
    seed: int = 0
    # training / held-out views
    n_train_views: int = 8
    train_azim_range: tuple = (0.0, 360.0)
    train_elev_range: tuple = (10.0, 50.0)
    n_test_views: int = 16
    test_elev_range: tuple = (5.0, 60.0)
    objects: list = field(default_factory=lambda: ["sphere", "box", "rods"])


# ---- surface samplers: return points (n,3), normals (n,3), colors (n,3) ------------------

def _stripes(u, v, freq, c0, c1):
    t = (torch.sin(freq * u) * torch.sin(freq * v) > 0).float()[:, None]
    return c0 * (1 - t) + c1 * t


def _sphere(n, g, center, radius, freq):
    d = torch.randn(n, 3, generator=g)
    d = d / d.norm(dim=-1, keepdim=True)
    phi = torch.atan2(d[:, 1], d[:, 0])
    th = torch.acos(d[:, 2].clamp(-1, 1))
    col = _stripes(phi, th * 2, freq * 0.5, torch.tensor([0.85, 0.2, 0.15]), torch.tensor([0.95, 0.85, 0.3]))
    return center + radius * d, d, col


def _box(n, g, center, half, freq):
    half = torch.as_tensor(half, dtype=torch.float32)
    areas = torch.stack([half[1] * half[2], half[0] * half[2], half[0] * half[1]]).repeat_interleave(2)
    face = torch.multinomial(areas, n, replacement=True, generator=g)
    axis, sign = face // 2, (face % 2) * 2 - 1
    uv = (torch.rand(n, 3, generator=g) * 2 - 1) * half
    pts = uv.clone()
    pts[torch.arange(n), axis] = sign.float() * half[axis]
    nrm = torch.zeros(n, 3)
    nrm[torch.arange(n), axis] = sign.float()
    others = torch.stack([uv[:, 0] + uv[:, 2], uv[:, 1] + uv[:, 2]], -1)
    col = _stripes(others[:, 0], others[:, 1], freq, torch.tensor([0.15, 0.35, 0.8]), torch.tensor([0.9, 0.9, 0.95]))
    return center + pts, nrm, col


def _cylinder(n, g, base, radius, height, color):
    a = torch.rand(n, generator=g) * 2 * math.pi
    h = torch.rand(n, generator=g) * height
    nrm = torch.stack([torch.cos(a), torch.sin(a), torch.zeros(n)], -1)
    pts = base + nrm * radius + torch.stack([torch.zeros(n), torch.zeros(n), h], -1)
    band = ((h * 12).floor() % 2)[:, None]
    col = color * (0.6 + 0.4 * band)
    return pts, nrm, col


def _sample_surfaces(n, g, cfg: SyntheticSceneConfig):
    parts = []
    # area weights so that density is roughly uniform
    specs = []
    if "sphere" in cfg.objects:
        specs.append(("sphere", 4 * math.pi * 0.38 ** 2))
    if "box" in cfg.objects:
        h = (0.28, 0.28, 0.35)
        specs.append(("box", 8 * (h[0] * h[1] + h[0] * h[2] + h[1] * h[2])))
    if "rods" in cfg.objects:
        specs.append(("rods", 3 * 2 * math.pi * 0.02 * 1.1))
    total = sum(a for _, a in specs)
    for name, area in specs:
        k = max(int(n * area / total), 500 if name == "rods" else 0)
        if name == "sphere":
            parts.append(_sphere(k, g, torch.tensor([-0.45, 0.0, 0.0]), 0.38, cfg.texture_freq))
        elif name == "box":
            parts.append(_box(k, g, torch.tensor([0.45, 0.3, -0.05]), (0.28, 0.28, 0.35), cfg.texture_freq))
        elif name == "rods":
            for j, (x, y) in enumerate([(0.15, -0.55), (0.35, -0.45), (0.55, -0.6)]):
                parts.append(_cylinder(k // 3, g, torch.tensor([x, y, -0.55]), 0.02, 1.1,
                                       torch.tensor([0.1, 0.6 + 0.1 * j, 0.25])))
    pts = torch.cat([p[0] for p in parts])
    nrm = torch.cat([p[1] for p in parts])
    col = torch.cat([p[2] for p in parts]).clamp(0.02, 0.98)
    return pts, nrm, col


def _basis_quats(normals: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) of a rotation whose third column is the normal."""
    n = normals / normals.norm(dim=-1, keepdim=True)
    helper = torch.where((n[:, 2].abs() < 0.9)[:, None], torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0]))
    t1 = torch.linalg.cross(helper, n)
    t1 = t1 / t1.norm(dim=-1, keepdim=True)
    t2 = torch.linalg.cross(n, t1)
    R = torch.stack([t1, t2, n], dim=-1)
    return _rotmat_to_quat(R)


def _rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    w = torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-12)) / 2
    x = torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12)) / 2
    y = torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-12)) / 2
    z = torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-12)) / 2
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q = torch.stack([w, x, y, z], -1)
    return q / q.norm(dim=-1, keepdim=True)


def build_gt_model(cfg: SyntheticSceneConfig) -> tuple[GaussianModel, torch.Tensor]:
    g = torch.Generator().manual_seed(cfg.seed)
    pts, nrm, col = _sample_surfaces(cfg.n_gt_gaussians, g, cfg)
    # tangent scale ~ sample spacing so that the surface is closed; thin along the normal
    area_per = _total_area(cfg) / pts.shape[0]
    s_t = 0.9 * math.sqrt(area_per)
    scales = torch.tensor([s_t, s_t, 0.1 * s_t]).expand(pts.shape[0], 3)
    model = GaussianModel(pts, torch.log(scales), _basis_quats(nrm),
                          inverse_sigmoid(torch.full((pts.shape[0],), 0.97)), inverse_sigmoid(col))
    ref_pts, _, _ = _sample_surfaces(cfg.n_ref_points, torch.Generator().manual_seed(cfg.seed + 1), cfg)
    return model, ref_pts


def _total_area(cfg):
    a = 0.0
    if "sphere" in cfg.objects:
        a += 4 * math.pi * 0.38 ** 2
    if "box" in cfg.objects:
        a += 8 * (0.28 * 0.28 + 0.28 * 0.35 + 0.28 * 0.35)
    if "rods" in cfg.objects:
        a += 3 * 2 * math.pi * 0.02 * 1.1
    return a


def make_views(cfg: SyntheticSceneConfig):
    r = cfg.resolution
    train = orbit_cameras(cfg.n_train_views, cfg.cam_radius, r, r, cfg.fov_deg,
                          azim_range=cfg.train_azim_range, elev_range=cfg.train_elev_range, prefix="train")
    # held-out views: full orbit, offset from training azimuths, different elevations
    test = orbit_cameras(cfg.n_test_views, cfg.cam_radius, r, r, cfg.fov_deg,
                         azim_range=(0.0, 360.0), elev_range=cfg.test_elev_range,
                         azim_offset=180.0 / cfg.n_test_views + 7.0, prefix="test")
    return train, test


@torch.no_grad()
def render_gt(gt_model: GaussianModel, cams: list[Camera], rasterize, bg, device):
    gt = GaussianModel.from_state_dict(gt_model.state_dict(), device)
    out = []
    for cam in cams:
        c = cam.to(device)
        img = gt.render(rasterize, c, bg)["image"].clamp(0, 1)
        c.image = img
        out.append(c)
    return out
