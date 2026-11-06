"""Benchmark-defined under-observation and geometric-failure labels."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class LabelParams:
    k_views: int = 2
    theta_deg: float = 10.0
    tau: float = 0.03
    min_contrib: float = 1e-2
    min_opacity: float = 0.05

    def __post_init__(self):
        if type(self.k_views) is not int or self.k_views < 0:
            raise ValueError("k_views must be a nonnegative integer")
        if not math.isfinite(self.theta_deg) or not 0 <= self.theta_deg <= 180:
            raise ValueError("theta_deg must be finite and in [0, 180]")
        for name in ("tau", "min_contrib", "min_opacity"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_opacity > 1:
            raise ValueError("min_opacity must not exceed 1")


@torch.no_grad()
def condition_labels(means: torch.Tensor, ref, train_cams, params: LabelParams,
                     gaussian_chunk_size: int = 512, ref_chunk_size: int = 8192) -> torch.Tensor:
    """Map each Gaussian to its nearest reference sample and label that sample's coverage.

    Visibility is an in-frustum and optional normal-facing proxy. It does not model
    depth occlusion. Pairwise view angles are computed in a vectorized (V,V,M) tensor.
    """
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if ref.points.ndim != 2 or ref.points.shape[-1] != 3 or ref.points.shape[0] == 0:
        raise ValueError("ref.points must be a non-empty (M, 3) tensor")
    if not train_cams:
        raise ValueError("train_cams must contain at least one camera")
    if gaussian_chunk_size <= 0 or ref_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if not torch.isfinite(means).all() or not torch.isfinite(ref.points).all():
        raise ValueError("Geometry must be finite")
    if params.k_views < 0 or not 0 <= params.theta_deg <= 180:
        raise ValueError("k_views must be nonnegative and theta_deg must be in [0, 180]")

    device = means.device
    ref_pts = ref.points.to(device=device, dtype=torch.float32)
    ref_normals = (ref.normals.to(device=device, dtype=torch.float32)
                   if ref.normals is not None else None)
    n_ref = ref_pts.shape[0]
    n_views = len(train_cams)
    visible = torch.zeros((n_views, n_ref), dtype=torch.bool, device=device)
    view_dirs = torch.zeros((n_views, n_ref, 3), dtype=torch.float32, device=device)

    for index, camera in enumerate(train_cams):
        R = camera.R.to(device=device, dtype=torch.float32)
        t = camera.t.to(device=device, dtype=torch.float32)
        center = camera.center.to(device=device, dtype=torch.float32)
        camera_points = ref_pts @ R.T + t
        z = camera_points[:, 2]
        safe_z = z.clamp_min(1e-5)
        u = camera.fx * camera_points[:, 0] / safe_z + camera.cx
        v = camera.fy * camera_points[:, 1] / safe_z + camera.cy
        in_frame = ((z > 0.01) & (u >= 0) & (u < camera.W) & (v >= 0) & (v < camera.H))

        direction_to_camera = center - ref_pts
        direction_to_camera /= direction_to_camera.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        if ref_normals is not None:
            front_facing = (ref_normals * direction_to_camera).sum(dim=-1) > 0.05
            in_frame &= front_facing
        visible[index] = in_frame
        view_dirs[index] = -direction_to_camera

    view_count = visible.sum(dim=0)
    # (V,V,M); invalid pairs are set to cosine 1 so they cannot determine the minimum.
    pair_cos = torch.einsum("vmd,wmd->vwm", view_dirs, view_dirs).clamp(-1.0, 1.0)
    valid_pairs = visible[:, None, :] & visible[None, :, :]
    pair_cos = torch.where(valid_pairs, pair_cos, torch.ones_like(pair_cos))
    min_cos = pair_cos.amin(dim=(0, 1))
    max_angle = torch.acos(min_cos)
    max_angle = torch.where(view_count >= 2, max_angle, torch.zeros_like(max_angle))
    ref_under_observed = ((view_count < params.k_views) |
                          (max_angle < math.radians(params.theta_deg)))

    n_gaussians = means.shape[0]
    result = torch.empty(n_gaussians, dtype=torch.bool, device=device)
    for start in range(0, n_gaussians, gaussian_chunk_size):
        chunk = means[start:start + gaussian_chunk_size].float()
        best_distance = torch.full((chunk.shape[0],), float("inf"), device=device)
        nearest_index = torch.zeros(chunk.shape[0], dtype=torch.long, device=device)
        for ref_start in range(0, n_ref, ref_chunk_size):
            ref_chunk = ref_pts[ref_start:ref_start + ref_chunk_size]
            block = torch.cdist(chunk, ref_chunk, p=2)
            block_distance, block_index = block.min(dim=1)
            update = block_distance < best_distance
            best_distance = torch.where(update, block_distance, best_distance)
            nearest_index = torch.where(update, block_index + ref_start, nearest_index)
        result[start:start + chunk.shape[0]] = ref_under_observed[nearest_index]
    return result


def outcome_labels(distance: torch.Tensor, heldout_contrib: torch.Tensor,
                   opacity: torch.Tensor, params: LabelParams,
                   apply_opacity_gate: bool = True) -> torch.Tensor:
    """Return f_i; optionally remove the opacity gate for a signal-independent D1 label."""
    if distance.shape != heldout_contrib.shape or distance.shape != opacity.shape:
        raise ValueError("distance, heldout_contrib, and opacity must have matching shapes")
    if distance.ndim != 1 or any(not torch.isfinite(x).all()
                                for x in (distance, heldout_contrib, opacity)):
        raise ValueError("Label inputs must be finite vectors")
    label = (distance > params.tau) & (heldout_contrib > params.min_contrib)
    if apply_opacity_gate:
        label &= opacity > params.min_opacity
    return label


def save_labels(path, xyz, distance, m, f, meta: dict, params: LabelParams):
    """Write geometric labels using the established archive format."""
    arrays = {
        "xyz": np.asarray(xyz, dtype=np.float32),
        "distance": np.asarray(distance, dtype=np.float32),
        "m_i": np.asarray(m, dtype=bool),
        "f_i": np.asarray(f, dtype=bool),
    }
    n = arrays["xyz"].shape[0]
    if arrays["xyz"].shape != (n, 3) or any(value.shape[0] != n for value in arrays.values()):
        raise ValueError("All label arrays must have the same N; xyz must have shape (N, 3)")
    if arrays["distance"].shape != (n,) or any(arrays[k].shape != (n,) for k in ("m_i", "f_i")):
        raise ValueError("distance, m_i, and f_i must have shape (N,)")
    if not np.isfinite(arrays["xyz"]).all() or not np.isfinite(arrays["distance"]).all():
        raise ValueError("Label geometry must be finite")
    metadata = dict(
        meta,
        label_params=asdict(params),
        m_i_available=True,
        provisional=True,
    )
    arrays["meta"] = np.array(metadata, dtype=object)
    np.savez_compressed(path, **arrays)
