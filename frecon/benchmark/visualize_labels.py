"""Figures for the geometric benchmark.                        Owner: Trường Thịnh

Required figures (docs/research/TruongThinh_geometric_failure_research.md):
  1. reference geometry + Gaussian centers (same frame)
  2. under-observed map (m_i)
  3. geometric failure map (f_i)
  4. continuous geometric error (d_i), e.g. color-coded centers or a histogram per label

TODO(TruongThinh):
  - [ ] 3D scatter projections (matplotlib, top/side views) — no extra dependency needed.
  - [ ] Optional: render Gaussians colored by label with the project rasterizer
        (model.render(..., features=<color per Gaussian>)) for a view-aligned figure.
  - [ ] Save under results/benchmark/figures/ (small PNGs only; see .gitignore).
"""
from __future__ import annotations


def plot_reference_and_gaussians(ref_points, means, out_path):
    raise NotImplementedError("TODO(TruongThinh)")


def plot_label_map(means, labels, title, out_path):
    raise NotImplementedError("TODO(TruongThinh)")


def plot_error(distance, m_i, f_i, out_path):
    raise NotImplementedError("TODO(TruongThinh)")
