"""Pinhole cameras (OpenCV convention: x right, y down, z forward) and view sampling."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch


@dataclass
class Camera:
    R: torch.Tensor  # (3, 3) world -> camera rotation
    t: torch.Tensor  # (3,)   world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    image: torch.Tensor | None = None  # (H, W, 3) in [0, 1]
    name: str = ""

    @property
    def center(self) -> torch.Tensor:
        return -self.R.T @ self.t

    def to(self, device) -> "Camera":
        img = None if self.image is None else self.image.to(device)
        return replace(self, R=self.R.to(device), t=self.t.to(device), image=img)


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor):
    f = target - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, up)
    if r.norm() < 1e-6:  # looking straight along `up`
        r = torch.linalg.cross(f, torch.tensor([1.0, 0.0, 0.0], dtype=eye.dtype))
    r = r / r.norm()
    d = torch.linalg.cross(f, r)  # image y axis points down
    R = torch.stack([r, d, f])
    t = -R @ eye
    return R, t


def focal_from_fov(fov_deg: float, size: int) -> float:
    return 0.5 * size / math.tan(0.5 * math.radians(fov_deg))


def orbit_cameras(n: int, radius: float, W: int, H: int, fov_deg: float,
                  azim_range=(0.0, 360.0), elev_range=(10.0, 50.0),
                  azim_offset: float = 0.0, prefix: str = "view") -> list[Camera]:
    """`n` cameras on a sphere of `radius` looking at the origin (z is up).

    Azimuths are evenly spaced in `azim_range`; elevations cycle through `elev_range`
    so that views are not coplanar.
    """
    a0, a1 = azim_range
    full_circle = abs((a1 - a0) - 360.0) < 1e-6
    if full_circle:
        azims = [a0 + azim_offset + i * 360.0 / n for i in range(n)]
    else:
        azims = [a0 + azim_offset + i * (a1 - a0) / max(n - 1, 1) for i in range(n)]
    e0, e1 = elev_range
    n_levels = 3 if n >= 3 else 1
    elevs = [e0 + (e1 - e0) * ((i % n_levels) / max(n_levels - 1, 1)) for i in range(n)]

    f = focal_from_fov(fov_deg, W)
    cams = []
    up = torch.tensor([0.0, 0.0, 1.0])
    for i, (az, el) in enumerate(zip(azims, elevs)):
        az_r, el_r = math.radians(az), math.radians(el)
        eye = torch.tensor([radius * math.cos(el_r) * math.cos(az_r),
                            radius * math.cos(el_r) * math.sin(az_r),
                            radius * math.sin(el_r)])
        R, t = look_at(eye, torch.zeros(3), up)
        cams.append(Camera(R=R, t=t, fx=f, fy=f, cx=W / 2, cy=H / 2, W=W, H=H,
                           name=f"{prefix}_{i:03d}"))
    return cams


def farthest_point_indices(points: torch.Tensor, k: int, start: int = 0) -> list[int]:
    """Greedy farthest-point sampling over camera centers (used for sparse-view subsets)."""
    k = min(k, points.shape[0])
    chosen = [start]
    d = (points - points[start]).norm(dim=-1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(d))
        chosen.append(nxt)
        d = torch.minimum(d, (points - points[nxt]).norm(dim=-1))
    return sorted(chosen)


def scene_extent(cams: list[Camera]) -> float:
    """3DGS 'nerf_normalization' radius: 1.1 x max distance of camera centers to their mean."""
    c = torch.stack([cam.center for cam in cams])
    return 1.1 * float((c - c.mean(0)).norm(dim=-1).max())
