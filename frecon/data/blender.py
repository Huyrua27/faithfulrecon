"""Loader for NeRF-Synthetic ("Blender") scenes: transforms_{train,test}.json + RGBA PNGs.

Cameras in the json are camera-to-world matrices in OpenGL convention (y up, -z forward);
they are converted to the OpenCV world-to-camera convention used by the rasterizer.
Images are composited on a white background and downsampled by `downscale`.

NeRF-Synthetic ships no reference geometry. D2.2's local Chamfer term is therefore
skipped unless a point cloud is given (`ref_points_path`, .ply/.npy/.pt), e.g. points
sampled from the .blend meshes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..camera import Camera, farthest_point_indices


def _load_split(root: Path, split: str, downscale: int, bg=(1.0, 1.0, 1.0)) -> list[Camera]:
    meta = json.loads((root / f"transforms_{split}.json").read_text())
    cams = []
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    for i, fr in enumerate(meta["frames"]):
        path = root / (fr["file_path"] + ("" if fr["file_path"].endswith(".png") else ".png"))
        im = Image.open(path)
        W0, H0 = im.size
        W, H = W0 // downscale, H0 // downscale
        im = im.resize((W, H), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:4] + np.array(bg, dtype=np.float32) * (1 - a[..., 3:4]) if a.shape[-1] == 4 else a
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
        Rc2w = c2w[:3, :3] @ flip  # OpenGL -> OpenCV camera axes
        R = Rc2w.T
        t = -R @ c2w[:3, 3]
        f = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
        cams.append(Camera(R=R, t=t, fx=f, fy=f, cx=W / 2, cy=H / 2, W=W, H=H,
                           image=torch.from_numpy(rgb), name=f"{split}_{i:03d}"))
    return cams


def load_blender(root: str, n_train_views: int, n_test_views: int, downscale: int = 4,
                 ref_points_path: str | None = None):
    root = Path(root)
    train_all = _load_split(root, "train", downscale)
    idx = farthest_point_indices(torch.stack([c.center for c in train_all]), n_train_views)
    train = [train_all[i] for i in idx]
    test_all = _load_split(root, "test", downscale)
    step = max(len(test_all) // n_test_views, 1)
    test = test_all[::step][:n_test_views]
    ref = _load_points(ref_points_path) if ref_points_path else None
    return train, test, ref


def _load_points(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".npy":
        return torch.from_numpy(np.load(p)).float()[:, :3]
    if p.suffix == ".pt":
        return torch.load(p).float()[:, :3]
    if p.suffix == ".ply":
        return _read_ply_xyz(p)
    raise ValueError(f"unsupported point file: {path}")


def _read_ply_xyz(path: Path) -> torch.Tensor:
    """Minimal PLY reader (ascii or binary_little_endian, vertex x/y/z as float)."""
    with open(path, "rb") as f:
        props, n, fmt, element = [], 0, "ascii", None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                element = line.split()[1]
                if element == "vertex":
                    n = int(line.split()[-1])
            elif line.startswith("property") and element == "vertex":
                props.append(line.split())
            elif line == "end_header":
                break
        names = [p[-1] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
        else:
            tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
                    "int": "<i4", "int32": "<i4", "ushort": "<u2", "short": "<i2"}
            dt = np.dtype([(p[-1], tmap[p[1]]) for p in props])
            rec = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            data = np.stack([rec[k].astype(np.float32) for k in names], -1)
    xyz = data[:, [names.index("x"), names.index("y"), names.index("z")]]
    return torch.from_numpy(np.ascontiguousarray(xyz)).float()
