"""Reference point cloud from a mesh by ray casting from the scene's cameras.

Used for NeRF-Synthetic, which ships no reference geometry. Sampling the first ray-surface hit from
the cameras keeps only surfaces that can be seen (e.g. the Lego model is made of many internal
bricks), the protocol used by prior work for Chamfer evaluation on this dataset.

The mesh must be in the scene's world frame. For NeRF-Synthetic that is the Blender world frame of
the .blend file; export it with scripts/blender_export_mesh.py (run inside Blender).

Usage (needs `pip install trimesh rtree`):
  python scripts/make_reference_points.py --config configs/blender_lego_v8.yaml \
      --mesh data/reference/lego.ply --out data/reference/lego_points.npy [--stride 2] [--max_points 300000]
Then set scene.ref_points_path to the .npy in the config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.data.blender import _load_split  # noqa: E402


def camera_rays(cam, stride: int):
    us = np.arange(0, cam.W, stride) + 0.5
    vs = np.arange(0, cam.H, stride) + 0.5
    u, v = np.meshgrid(us, vs)
    d_cam = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], -1).reshape(-1, 3)
    R = cam.R.numpy().astype(np.float64)
    d = d_cam @ R                                   # R^T applied row-wise
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(cam.center.numpy().astype(np.float64), d.shape)
    return o, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--stride", type=int, default=2, help="pixel stride at the config's downscale")
    ap.add_argument("--max_points", type=int, default=300_000)
    args = ap.parse_args()

    import trimesh

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sc = cfg["scene"]
    mesh = trimesh.load(args.mesh, force="mesh")
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, bounds {mesh.bounds.tolist()}")
    hits = []
    for split in args.splits:
        cams = _load_split(Path(sc["root"]), split, sc.get("downscale", 4))
        for i, cam in enumerate(cams):
            o, d = camera_rays(cam, args.stride)
            loc, _, _ = mesh.ray.intersects_location(o, d, multiple_hits=False)
            hits.append(loc)
            if i % 20 == 0:
                print(f"  {split} {i}/{len(cams)}: {len(loc)} hits")
    pts = np.concatenate(hits).astype(np.float32)
    if len(pts) > args.max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, pts)
    print(f"saved {len(pts)} points -> {args.out}")


if __name__ == "__main__":
    main()
