"""DTU loader (IDR / NeuS / 2DGS preprocessed format) with the official reference point cloud.

Scan directory (e.g. data/dtu/scan24/):
    image/000000.png ...            RGB images (1600x1200)
    mask/000.png ...                object masks (same order as images)
    cameras.npz | cameras_sphere.npz
        world_mat_{i}  4x4, top 3x4 = P = K [R | t] in DTU world coordinates (OpenCV axes)
        scale_mat_{i}  4x4, maps the normalized frame (object ~ unit sphere) to DTU world

We work in the NORMALIZED frame: P_n = (world_mat @ scale_mat)[:3, :4], decomposed into K, R, t.
Images are composited on a white background with the masks (object-only reconstruction), so that
the reference geometry covers what is reconstructed.

Reference geometry: the official STL point cloud (Points/stl/stl{scan:03d}_total.ply from the DTU
MVS 2014 release), mapped to the normalized frame with inv(scale_mat) and cropped to points that
project inside the object mask of at least `min_mask_views` views. The cropped cloud is cached as .npy
next to the scan (`ref_cache`). Note: the DTU reference contains outliers (see DTU literature).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import rq

from ..camera import Camera, farthest_point_indices


def decompose_projection(P: np.ndarray):
    """P (3x4) = K [R | t]  ->  K (3x3, positive diagonal, K[2,2]=1), R (proper rotation), t (3,)."""
    M = P[:, :3]
    K, R = rq(M)
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R            # make K's diagonal positive (D @ D = I)
    t = np.linalg.solve(K, P[:, 3])
    if np.linalg.det(R) < 0:       # P is only defined up to scale (sign included)
        R, t = -R, -t
    K = K / K[2, 2]
    return K, R, t


def _load_cameras(scan: Path):
    for name in ("cameras_sphere.npz", "cameras.npz"):
        if (scan / name).exists():
            return np.load(scan / name), name
    raise FileNotFoundError(f"no cameras_sphere.npz / cameras.npz in {scan}")


def load_dtu_views(scan_dir: str, downscale: int = 4, bg=(1.0, 1.0, 1.0)):
    scan = Path(scan_dir)
    cams_npz, _ = _load_cameras(scan)
    imgs = sorted(glob.glob(str(scan / "image" / "*.png")))
    masks = sorted(glob.glob(str(scan / "mask" / "*.png")))
    if len(masks) != len(imgs):
        raise ValueError(f"{len(imgs)} images but {len(masks)} masks in {scan}")
    cams, mask_arrays = [], []
    scale_mat = None
    for i, (ip, mp) in enumerate(zip(imgs, masks)):
        world_mat = cams_npz[f"world_mat_{i}"].astype(np.float64)
        scale_mat = cams_npz[f"scale_mat_{i}"].astype(np.float64)
        K, R, t = decompose_projection((world_mat @ scale_mat)[:3, :4])
        im = Image.open(ip).convert("RGB")
        W0, H0 = im.size
        W, H = W0 // downscale, H0 // downscale
        rgb = np.asarray(im.resize((W, H), Image.LANCZOS), dtype=np.float32) / 255.0
        m = np.asarray(Image.open(mp).convert("L").resize((W, H), Image.NEAREST), dtype=np.float32) / 255.0
        m = (m > 0.5).astype(np.float32)[..., None]
        rgb = rgb * m + np.asarray(bg, np.float32) * (1 - m)
        sx, sy = W / W0, H / H0
        # OpenCV pixel centers are at integer coordinates; ours are at index + 0.5
        fx, fy = K[0, 0] * sx, K[1, 1] * sy
        cx, cy = (K[0, 2] + 0.5) * sx, (K[1, 2] + 0.5) * sy
        cams.append(Camera(R=torch.tensor(R, dtype=torch.float32), t=torch.tensor(t, dtype=torch.float32),
                           fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), W=W, H=H,
                           image=torch.from_numpy(rgb), name=f"dtu_{i:03d}"))
        mask_arrays.append(m[..., 0] > 0.5)
    return cams, mask_arrays, scale_mat


def split_views(cams, n_train: int, n_test: int):
    """Sparse training views by farthest-point sampling of camera centers; held-out views evenly
    spaced among the remaining ones."""
    centers = torch.stack([c.center for c in cams])
    train_idx = farthest_point_indices(centers, n_train)
    rest = [i for i in range(len(cams)) if i not in train_idx]
    step = max(len(rest) // max(n_test, 1), 1)
    test_idx = rest[::step][:n_test]
    return train_idx, test_idx


def reference_points(stl_ply: str, scale_mat: np.ndarray, cams, masks, min_mask_views: int = 1,
                     max_points: int = 200_000, cache: str | None = None, seed: int = 0) -> torch.Tensor:
    if cache and Path(cache).exists():
        return torch.from_numpy(np.load(cache)).float()
    from .blender import _read_ply_xyz
    pts_w = _read_ply_xyz(Path(stl_ply)).numpy().astype(np.float64)
    inv = np.linalg.inv(scale_mat)
    pts = (pts_w @ inv[:3, :3].T) + inv[:3, 3]
    pts = pts[np.linalg.norm(pts, axis=1) < 1.5]           # drop far-away table / clutter early
    count = np.zeros(len(pts), dtype=np.int32)
    for cam, m in zip(cams, masks):
        R, t = cam.R.numpy().astype(np.float64), cam.t.numpy().astype(np.float64)
        pc = pts @ R.T + t
        z = pc[:, 2]
        ok = z > 1e-6
        u = np.floor(cam.fx * pc[:, 0] / np.where(ok, z, 1) + cam.cx).astype(np.int64)
        v = np.floor(cam.fy * pc[:, 1] / np.where(ok, z, 1) + cam.cy).astype(np.int64)
        inb = ok & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        hit = np.zeros(len(pts), dtype=bool)
        hit[inb] = m[v[inb], u[inb]]
        count += hit
    pts = pts[count >= min_mask_views]
    if len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    pts = pts.astype(np.float32)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, pts)
    return torch.from_numpy(pts)


def load_dtu(scan_dir: str, n_train_views: int, n_test_views: int, downscale: int = 4,
             stl_ply: str | None = None, ref_cache: str | None = None, min_mask_views: int = 1,
             max_ref_points: int = 200_000):
    cams, masks, scale_mat = load_dtu_views(scan_dir, downscale)
    tr, te = split_views(cams, n_train_views, n_test_views)
    ref = None
    if stl_ply or (ref_cache and Path(ref_cache).exists()):
        ref = reference_points(stl_ply, scale_mat, cams, masks, min_mask_views, max_ref_points, ref_cache)
    return [cams[i] for i in tr], [cams[i] for i in te], ref, {"train_idx": tr, "test_idx": te}
