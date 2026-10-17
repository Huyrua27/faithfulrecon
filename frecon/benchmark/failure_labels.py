"""Benchmark-defined labels m_i (condition) and f_i (outcome).   Owner: Trường Thịnh"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import numpy as np
import torch
from .reference_geometry import ReferenceGeometry


@dataclass
class LabelParams:
    k_views: int = 2            # m_i: nhìn thấy dưới k training views
    theta_deg: float = 10.0     # m_i: góc tam giác đạc cực đại dưới theta độ
    tau: float = 0.03           # f_i: khoảng cách vượt quá tau
    min_contrib: float = 1e-2   # f_i: đóng góp alpha trên test views
    min_opacity: float = 0.05   # f_i: opacity tối thiểu


def condition_labels(means: torch.Tensor, ref: ReferenceGeometry, train_cams, params: LabelParams) -> torch.Tensor:
    """Compute under-observation labels from frustum coverage,
    optional normal-facing checks, and view-angle criteria.
    No explicit depth-based occlusion test is performed."""
    device = means.device
    ref_pts = ref.points.to(device=device, dtype=torch.float32)
    ref_nrm = ref.normals.to(device=device, dtype=torch.float32) if ref.normals is not None else None
    M = ref_pts.shape[0]
    V = len(train_cams)

    vis_mask = torch.zeros((V, M), dtype=torch.bool, device=device)
    view_dirs = torch.zeros((V, M, 3), dtype=torch.float32, device=device)

    for v_idx, cam in enumerate(train_cams):
        R = cam.R.to(device)
        t = cam.t.to(device)
        cam_center = cam.center.to(device)

        pts_cam = (ref_pts @ R.T) + t
        z = pts_cam[:, 2]
        in_front = z > 0.01

        safe_z = torch.clamp(z, min=1e-5)
        u = cam.fx * (pts_cam[:, 0] / safe_z) + cam.cx
        v = cam.fy * (pts_cam[:, 1] / safe_z) + cam.cy
        in_frustum = in_front & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)

        ray_to_cam = cam_center - ref_pts
        ray_to_cam = ray_to_cam / torch.clamp(torch.norm(ray_to_cam, dim=-1, keepdim=True), min=1e-7)

        # Mặt trước hướng về camera
        if ref_nrm is not None:
            facing_cam = (ref_nrm * ray_to_cam).sum(dim=-1) > 0.05
            visible = in_frustum & facing_cam
        else:
            visible = in_frustum

        vis_mask[v_idx] = visible
        view_dirs[v_idx] = -ray_to_cam

    view_count = vis_mask.sum(dim=0)
    theta_rad = math.radians(params.theta_deg)
    max_angles = torch.zeros(M, dtype=torch.float32, device=device)

    for j in range(M):
        seen_views = torch.where(vis_mask[:, j])[0]
        if len(seen_views) >= 2:
            rays = view_dirs[seen_views, j]
            cos_matrix = torch.matmul(rays, rays.T).clamp(-1.0, 1.0)
            min_cos = cos_matrix.min()
            max_angles[j] = torch.acos(min_cos)
        else:
            max_angles[j] = 0.0

    ref_under_observed = (view_count < params.k_views) | (max_angles < theta_rad)

    # Ánh xạ tối ưu bộ nhớ: Map each Gaussian center to its nearest sampled reference point.
    N = means.shape[0]
    m_i = torch.zeros(N, dtype=torch.bool, device=device)
    gaussian_chunk_size = 512
    ref_chunk_size = 8192

    for i in range(0, N, gaussian_chunk_size):
        chunk = means[i : i + gaussian_chunk_size]
        best_dist = torch.full((chunk.shape[0],), float("inf"), device=device, dtype=torch.float32)
        nearest_idx = torch.zeros(chunk.shape[0], device=device, dtype=torch.long)

        for j in range(0, M, ref_chunk_size):
            ref_chunk = ref_pts[j : j + ref_chunk_size]
            block_dist = torch.cdist(chunk, ref_chunk, p=2)
            block_min, block_idx = block_dist.min(dim=1)
            update = block_min < best_dist
            best_dist = torch.where(update, block_min, best_dist)
            nearest_idx = torch.where(update, block_idx + j, nearest_idx)

        m_i[i : i + chunk.shape[0]] = ref_under_observed[nearest_idx]

    return m_i


def outcome_labels(distance: torch.Tensor, heldout_contrib: torch.Tensor, opacity: torch.Tensor,
                   params: LabelParams) -> torch.Tensor:
    far = distance > params.tau
    visible_on_test = heldout_contrib > params.min_contrib
    sufficient_opacity = opacity > params.min_opacity
    return far & visible_on_test & sufficient_opacity


def save_labels(path, xyz, distance, m, f, meta: dict, params: LabelParams):
    meta = dict(
        meta,
        label_params=asdict(params),
        m_i_available=True,
        provisional=True,
        validation_status="basic_checks_passed_geometry_assumptions_unverified",
        known_limitations=[
            "distance is to the nearest sampled reference point",
            "visibility uses frustum and optional normal-facing checks",
            "no explicit depth-based occlusion test",
            "full reference-to-model alignment is not verified",
        ],
    )
    np.savez_compressed(
        path,
        xyz=np.asarray(xyz, dtype=np.float32),
        distance=np.asarray(distance, dtype=np.float32),
        m_i=np.asarray(m, dtype=bool),
        f_i=np.asarray(f, dtype=bool),
        meta=np.array(meta, dtype=object),
    )
