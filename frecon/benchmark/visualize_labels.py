"""Small diagnostic figures for geometric benchmark labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    _pyplot().close(fig)


def _equal_3d(ax, points):
    points = np.asarray(points)
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()) / 2, 1e-3) * 1.03
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def plot_reference_and_gaussians(ref_points, means, out_path):
    plt = _pyplot()
    ref_points, means = np.asarray(ref_points), np.asarray(means)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*ref_points.T, s=0.35, c="#8793a0", alpha=0.16,
               label=f"Reference ({len(ref_points):,})", rasterized=True)
    ax.scatter(*means.T, s=0.45, c="#d95f4f", alpha=0.20,
               label=f"Gaussian centers ({len(means):,})", rasterized=True)
    _equal_3d(ax, np.concatenate((ref_points, means), axis=0))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Reference geometry and Gaussian centers")
    ax.legend(loc="upper left")
    _finish(fig, out_path)


def plot_label_map(means, labels, title, out_path):
    plt = _pyplot()
    means, labels = np.asarray(means), np.asarray(labels, dtype=bool)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*means[~labels].T, s=0.45, c="#aab2bd", alpha=0.12,
               label=f"Not selected ({int((~labels).sum()):,})", rasterized=True)
    if labels.any():
        ax.scatter(*means[labels].T, s=1.3, c="#d84032", alpha=0.80,
                   label=f"Selected ({int(labels.sum()):,})", rasterized=True)
    _equal_3d(ax, means)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=title.replace("_", " "))
    ax.legend(loc="upper left")
    _finish(fig, out_path)


def plot_error(distance, m_i, f_i, out_path):
    plt = _pyplot()
    distance = np.asarray(distance, dtype=np.float32)
    m_i, f_i = np.asarray(m_i, dtype=bool), np.asarray(f_i, dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(distance[~f_i], bins=60, alpha=0.7, label=f"f_i=0 (n={int((~f_i).sum()):,})")
    if f_i.any():
        axes[0].hist(distance[f_i], bins=60, alpha=0.7, label=f"f_i=1 (n={int(f_i.sum()):,})")
    axes[0].set(xlabel="Nearest-reference distance", ylabel="Gaussian count", title="Distance by outcome label")
    axes[0].legend()
    if m_i.any() and (~m_i).any():
        axes[1].boxplot([distance[~m_i], distance[m_i]], labels=["m_i=0", "m_i=1"], showfliers=False)
    else:
        axes[1].hist(distance[m_i], bins=60, alpha=0.7, label=f"m_i=1 (n={int(m_i.sum()):,})")
        axes[1].hist(distance[~m_i], bins=60, alpha=0.7, label=f"m_i=0 (n={int((~m_i).sum()):,})")
        axes[1].legend()
    axes[1].set(ylabel="Nearest-reference distance", title="Distance by condition label")
    fig.tight_layout()
    _finish(fig, out_path)
