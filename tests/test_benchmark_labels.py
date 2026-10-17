"""Benchmark label tests.                                         Owner: Trường Thịnh"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from frecon.benchmark.failure_labels import LabelParams, condition_labels, outcome_labels, save_labels
from frecon.benchmark.geometric_error import point_to_surface_distance
from frecon.benchmark.reference_geometry import ReferenceGeometry


def test_distance_on_sphere():
    """Centers placed at known radii around a sampled sphere -> |d_i - (r_i - R)| < spacing."""
    R = 1.0
    num_samples = 4000
    phi = np.random.uniform(0, 2 * np.pi, num_samples)
    cos_theta = np.random.uniform(-1, 1, num_samples)
    sin_theta = np.sqrt(np.maximum(0.0, 1 - cos_theta**2))
    x = R * sin_theta * np.cos(phi)
    y = R * sin_theta * np.sin(phi)
    z = R * cos_theta
    sphere_pts = np.stack([x, y, z], axis=-1).astype(np.float32)

    ref = ReferenceGeometry(
        points=torch.from_numpy(sphere_pts),
        normals=torch.from_numpy(sphere_pts / R),
    )

    radii = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5], dtype=torch.float32)
    query_dirs = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7071, 0.7071, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    means = query_dirs * radii[:, None]

    dists = point_to_surface_distance(means, ref)
    expected_dists = torch.abs(radii - R)

    assert torch.all(torch.abs(dists - expected_dists) <= 0.06)


def test_condition_label_ignores_model():
    """m_i must not change when the Gaussian model is perturbed but cameras/reference are fixed
    (only through which reference point is nearest)."""
    class DummyCam:
        def __init__(self, center):
            self.R = torch.eye(3, dtype=torch.float32)
            self.t = -center
            self.center = center
            self.fx = 100.0
            self.fy = 100.0
            self.cx = 50.0
            self.cy = 50.0
            self.W = 100
            self.H = 100

    ref_pts = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    ref = ReferenceGeometry(points=ref_pts, normals=torch.tensor([[0.0, 0.0, -1.0]]))
    cams = [DummyCam(torch.tensor([0.0, 0.0, 0.0]))]
    params = LabelParams(k_views=2, theta_deg=10.0)

    means1 = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    means2 = torch.tensor([[0.001, 0.001, 2.001]], dtype=torch.float32)

    m1 = condition_labels(means1, ref, cams, params)
    m2 = condition_labels(means2, ref, cams, params)
    assert torch.equal(m1, m2)


def test_outcome_label_thresholds():
    """f_i is monotone in tau: increasing tau never adds positives."""
    N = 100
    distance = torch.linspace(0.0, 0.1, N)
    heldout_contrib = torch.full((N,), 0.05)
    opacity = torch.full((N,), 0.8)

    tau_small = 0.02
    tau_large = 0.05

    f_small = outcome_labels(distance, heldout_contrib, opacity, LabelParams(tau=tau_small))
    f_large = outcome_labels(distance, heldout_contrib, opacity, LabelParams(tau=tau_large))

    assert torch.all(f_small[f_large] == True)
    assert f_small.sum() >= f_large.sum()


def test_labels_file_roundtrip(tmp_path):
    """save_labels -> np.load gives the keys/dtypes of docs/data_format.md §4."""
    N = 10
    xyz = np.zeros((N, 3), dtype=np.float32)
    distance = np.ones(N, dtype=np.float32) * 0.05
    m = np.zeros(N, dtype=bool)
    f = np.ones(N, dtype=bool)
    meta = {"scene": "synthetic", "seed": 0}
    params = LabelParams()

    out_file = tmp_path / "geometric_labels.npz"
    save_labels(out_file, xyz, distance, m, f, meta, params)

    assert out_file.exists()
    loaded = np.load(out_file, allow_pickle=True)

    required_keys = {"xyz", "distance", "m_i", "f_i", "meta"}
    assert required_keys.issubset(set(loaded.files))

    assert loaded["xyz"].shape == (N, 3)
    assert loaded["xyz"].dtype == np.float32
    assert loaded["distance"].shape == (N,)
    assert loaded["distance"].dtype == np.float32
    assert loaded["m_i"].dtype == bool
    assert loaded["f_i"].dtype == bool

    meta_loaded = loaded["meta"].item()
    assert meta_loaded["scene"] == "synthetic"
    assert meta_loaded["m_i_available"] is True
    assert meta_loaded["provisional"] is True
