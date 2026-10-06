"""Reference geometry loading and alignment.                    Owner: Trường Thịnh

The synthetic scene (frecon/data/synthetic.py) already returns reference surface points in
the SAME world frame as the Gaussian model (no transform needed). NeRF-Synthetic has no
geometry: points must be sampled from the .blend meshes and brought into the frame used
by transforms_*.json.

TODO(TruongThinh):
  - [ ] load_reference(): synthetic -> build_gt_model(cfg)[1]; blender -> point file
        (.ply/.npy/.pt; a reader exists in frecon/data/blender.py::_load_points).
  - [ ] Return surface normals too if available (useful for point-to-plane distance).
  - [ ] check_alignment(): render the reference points into a training view and compare with
        the GT alpha mask / image silhouette; report the misalignment in pixels.
        Must pass before any label is trusted.
  - [ ] Document units and coordinate frame in docs/research/TruongThinh_geometric_failure_research.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReferenceGeometry:
    points: torch.Tensor                 # (M, 3) world frame of the Gaussian model
    normals: torch.Tensor | None = None  # (M, 3) optional
    source: str = ""                     # e.g. "synthetic:seed0" or path to the point file


def load_reference(scene_cfg: dict) -> ReferenceGeometry:
    raise NotImplementedError("TODO(TruongThinh): load reference geometry for the scene config")


def check_alignment(ref: ReferenceGeometry, cams, rasterize=None) -> dict:
    """Return per-view silhouette agreement (e.g. IoU, mean pixel offset). TODO(TruongThinh)."""
    raise NotImplementedError("TODO(TruongThinh): alignment sanity check")
