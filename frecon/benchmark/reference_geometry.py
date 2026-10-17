"""Reference geometry loading and alignment.                    Owner: Trường Thịnh"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class ReferenceGeometry:
    points: torch.Tensor                 # (M, 3) world frame của Gaussian model
    normals: torch.Tensor | None = None  # (M, 3) pháp tuyến (nếu có)
    source: str = ""                     # Nguồn dữ liệu


def load_reference(scene_cfg: dict) -> ReferenceGeometry:
    scene_type = scene_cfg.get("type", "synthetic")

    if scene_type == "synthetic":
        from frecon.data.synthetic import SyntheticSceneConfig, _sample_surfaces
        cfg = SyntheticSceneConfig(
            resolution=scene_cfg.get("resolution", 160),
            n_ref_points=scene_cfg.get("n_ref_points", 60000),
            seed=scene_cfg.get("seed", 0)
        )
        g = torch.Generator().manual_seed(cfg.seed + 1)
        pts, nrm, _ = _sample_surfaces(cfg.n_ref_points, g, cfg)
        return ReferenceGeometry(points=pts, normals=nrm, source="synthetic")

    elif scene_type == "blender":
        from frecon.data.blender import _load_points
        root = scene_cfg.get("root", "")
        pts = _load_points(root)
        return ReferenceGeometry(points=pts, source=f"blender:{root}")

    raise ValueError(f"Unknown scene type: {scene_type}")


@torch.no_grad()
def check_alignment(ref: ReferenceGeometry, cams, rasterize=None) -> dict:
    """Sanity check: measure reference-point frustum coverage only.
    This does NOT prove that reference geometry and the Gaussian model
    are aligned in scale, orientation, or translation.
    """
    if len(cams) == 0:
        raise ValueError("Cannot check frustum coverage: no cameras")
    ref_pts = ref.points
    total_points = ref_pts.shape[0]
    if total_points == 0:
        raise ValueError("Reference geometry contains no points")

    total_in_view = 0
    for cam in cams:
        dev = cam.R.device
        pts = ref_pts.to(device=dev, dtype=torch.float32)
        pts_cam = (pts @ cam.R.T) + cam.t
        z = pts_cam[:, 2]
        in_front = z > 0
        safe_z = torch.clamp(z, min=1e-5)
        u = cam.fx * (pts_cam[:, 0] / safe_z) + cam.cx
        v = cam.fy * (pts_cam[:, 1] / safe_z) + cam.cy
        in_view = (
            in_front
            & (u >= 0) & (u < cam.W)
            & (v >= 0) & (v < cam.H)
        )
        total_in_view += int(in_view.sum().item())

    avg_visible_per_cam = total_in_view / len(cams)
    return {
        "frustum_coverage_avg_pts": avg_visible_per_cam,
        "coverage_ratio": avg_visible_per_cam / total_points,
        "total_points": total_points,
        "note": "frustum_coverage_only_not_verified_alignment"
    }
