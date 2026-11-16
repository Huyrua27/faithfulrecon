"""Sparse points for SfM-style initialization from the CLEAN training views with known poses.

Replaces COLMAP's point_triangulator (features + matching + triangulation with fixed poses):
SIFT inside the object masks -> ratio-test matching for every view pair -> epipolar check with the
known fundamental matrix -> DLT triangulation -> cheirality, reprojection and triangulation-angle checks
-> voxel de-duplication. Positions are identical for every photometric level of a (scan, N) pair, so
paired comparisons isolate the effect of the perturbation on 3DGS optimization ("clean-SfM" init).

python scripts/sfm_tri.py --root data/dtu/scan63 --n_train 8 --out data/dtu/scan63/sfm_v8.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def fundamental(K1, R1, t1, K2, R2, t2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    return np.linalg.inv(K2).T @ (_skew(t) @ R) @ np.linalg.inv(K1)


def sampson(F, x1, x2):
    h1 = np.hstack([x1, np.ones((len(x1), 1))])
    h2 = np.hstack([x2, np.ones((len(x2), 1))])
    Fx1, Ftx2 = h1 @ F.T, h2 @ F
    num = np.sum(h2 * Fx1, 1) ** 2
    return num / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12)


def triangulate_views(images, masks, Ks, Rs, ts, ratio=0.75, max_epi=1.5, max_reproj=1.5,
                      min_angle_deg=2.0, voxel=0.004, n_features=8000):
    """images: (H, W, 3) float in [0, 1]; masks: (H, W) bool; K, R, t in the convention of frecon.camera
    (pixel centres at index + 0.5). Returns (M, 3) float32 points."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    feats = []
    for img, m in zip(images, masks):
        g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(g, m.astype(np.uint8) * 255)
        xy = np.float64([k.pt for k in kp]).reshape(-1, 2) + 0.5     # OpenCV pixel centres -> ours
        feats.append((xy, des))
    P = [K @ np.hstack([R, t[:, None]]) for K, R, t in zip(Ks, Rs, ts)]
    C = [-R.T @ t for R, t in zip(Rs, ts)]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out, n = [], len(images)
    for i in range(n):
        for j in range(i + 1, n):
            (xi, di), (xj, dj) = feats[i], feats[j]
            if di is None or dj is None or len(di) < 2 or len(dj) < 2:
                continue
            good = [a for a, b in (p for p in bf.knnMatch(di, dj, k=2) if len(p) == 2) if a.distance < ratio * b.distance]
            if len(good) < 8:
                continue
            x1 = xi[[g.queryIdx for g in good]]
            x2 = xj[[g.trainIdx for g in good]]
            F = fundamental(Ks[i], Rs[i], ts[i], Ks[j], Rs[j], ts[j])
            ok = sampson(F, x1, x2) < max_epi ** 2
            x1, x2 = x1[ok], x2[ok]
            if len(x1) == 0:
                continue
            X = cv2.triangulatePoints(P[i], P[j], x1.T, x2.T)
            X = (X[:3] / X[3]).T
            keep = np.ones(len(X), bool)
            for Pk, xk, Rk, tk in ((P[i], x1, Rs[i], ts[i]), (P[j], x2, Rs[j], ts[j])):
                z = X @ Rk[2] + tk[2]
                pr = np.hstack([X, np.ones((len(X), 1))]) @ Pk.T
                err = np.linalg.norm(pr[:, :2] / pr[:, 2:3] - xk, axis=1)
                keep &= (z > 1e-6) & (err < max_reproj)
            r1, r2 = X - C[i], X - C[j]
            cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
            keep &= cosang < np.cos(np.radians(min_angle_deg))
            out.append(X[keep])
    if not out:
        return np.zeros((0, 3), np.float32)
    X = np.concatenate(out)
    X = X[np.linalg.norm(X, axis=1) < 1.5]
    _, idx = np.unique(np.floor(X / voxel).astype(np.int64), axis=0, return_index=True)
    return X[np.sort(idx)].astype(np.float32)


def scan_points(root: str, n_train: int, downscale: int = 2):
    """Triangulate from the FPS training views of a DTU scan (same views as training, clean images)."""
    from frecon.camera import farthest_point_indices
    from frecon.data.dtu import load_dtu_views
    cams, masks, _ = load_dtu_views(root, downscale)
    tr = farthest_point_indices(torch.stack([c.center for c in cams]), n_train)
    Ks = [np.array([[cams[i].fx, 0, cams[i].cx], [0, cams[i].fy, cams[i].cy], [0, 0, 1]]) for i in tr]
    return triangulate_views([cams[i].image.numpy() for i in tr], [masks[i] for i in tr], Ks,
                             [cams[i].R.numpy().astype(np.float64) for i in tr],
                             [cams[i].t.numpy().astype(np.float64) for i in tr])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()
    pts = scan_points(a.root, a.n_train, a.downscale)
    np.save(a.out, pts)
    print(f"{a.root}: {len(pts)} points -> {a.out}")
