"""Load reference samples in the same coordinate frame as the Gaussian model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ReferenceGeometry:
    points: torch.Tensor
    normals: torch.Tensor | None = None
    source: str = ""


def load_reference(scene_cfg: dict, reference_points: torch.Tensor | None = None) -> ReferenceGeometry:
    """Load geometry using the complete scene definition or a scene-loader result.

    Prefer passing `reference_points` returned by `frecon.data.load_scene`; this ensures
    the labels use the exact same geometry as the camera/data pipeline.
    """
    if reference_points is not None:
        return ReferenceGeometry(reference_points, source=str(scene_cfg.get("root", scene_cfg.get("type", ""))))

    scene_type = scene_cfg.get("type", "synthetic")
    if scene_type == "synthetic":
        from frecon.data.synthetic import SyntheticSceneConfig, _sample_surfaces

        allowed = SyntheticSceneConfig.__dataclass_fields__.keys()
        values = {
            key: (tuple(value) if isinstance(value, list) and key.endswith("range") else value)
            for key, value in scene_cfg.items() if key in allowed
        }
        config = SyntheticSceneConfig(**values)
        generator = torch.Generator().manual_seed(config.seed + 1)
        points, normals, _ = _sample_surfaces(config.n_ref_points, generator, config)
        return ReferenceGeometry(points=points, normals=normals, source="synthetic:configured_scene")

    if scene_type == "blender":
        ref_path = scene_cfg.get("ref_points_path")
        if not ref_path:
            raise ValueError("Blender reference geometry requires scene.ref_points_path")
        from frecon.data.blender import _load_points

        path = Path(ref_path)
        if not path.is_file():
            raise FileNotFoundError(f"Blender reference point file does not exist: {path}")
        return ReferenceGeometry(points=_load_points(str(path)), source=f"blender:{path}")

    if scene_type == "dtu":
        from frecon.data.dtu import load_dtu

        _, _, points, _ = load_dtu(
            scene_cfg["root"], scene_cfg["n_train_views"], scene_cfg["n_test_views"],
            scene_cfg.get("downscale", 4), scene_cfg.get("stl_ply"),
            scene_cfg.get("ref_cache"), scene_cfg.get("min_mask_views", 1),
            scene_cfg.get("max_ref_points", 200_000),
        )
        if points is None:
            raise ValueError("DTU reference geometry is unavailable; set stl_ply or ref_cache")
        return ReferenceGeometry(points=points, source=f"dtu:{scene_cfg['root']}")

    raise ValueError(f"Unknown scene type: {scene_type}")


@torch.no_grad()
def check_alignment(ref: ReferenceGeometry, cams, rasterize=None) -> dict:
    """Report reference point frustum coverage; this is not a geometric alignment test."""
    if not cams:
        raise ValueError("Cannot check frustum coverage: no cameras")
    if ref.points.ndim != 2 or ref.points.shape[0] == 0:
        raise ValueError("Reference geometry must contain points with shape (M, 3)")

    total_points = ref.points.shape[0]
    total_in_view = 0
    per_view = []
    for cam in cams:
        device = cam.R.device
        points = ref.points.to(device=device, dtype=torch.float32)
        points_cam = points @ cam.R.T + cam.t
        z = points_cam[:, 2]
        safe_z = z.clamp_min(1e-5)
        u = cam.fx * points_cam[:, 0] / safe_z + cam.cx
        v = cam.fy * points_cam[:, 1] / safe_z + cam.cy
        in_view = ((z > 0) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H))
        count = int(in_view.sum().item())
        total_in_view += count
        per_view.append(count)
    return {
        "frustum_coverage_avg_points_per_view": total_in_view / len(cams),
        "frustum_coverage_ratio_avg_per_view": total_in_view / (len(cams) * total_points),
        "per_view_points": per_view,
        "total_reference_points": total_points,
        "note": "frustum coverage only; scale/orientation/translation alignment is unverified",
    }
