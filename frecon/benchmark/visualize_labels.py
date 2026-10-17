"""Figures for the geometric benchmark.                        Owner: Trường Thịnh"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_reference_and_gaussians(ref_points, means, out_path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    r_sub = ref_points[::10]
    g_sub = means[::10]
    ax.scatter(r_sub[:, 0], r_sub[:, 1], r_sub[:, 2], c="gray", alpha=0.3, s=1, label="Reference Surface")
    ax.scatter(g_sub[:, 0], g_sub[:, 1], g_sub[:, 2], c="red", alpha=0.5, s=2, label="Gaussians")
    ax.set_title("Reference Geometry vs Gaussian Primitives")
    ax.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_label_map(means, labels, title, out_path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    pos = labels.astype(bool)
    neg = ~pos
    if np.any(neg):
        ax.scatter(means[neg, 0], means[neg, 1], means[neg, 2], c="blue", alpha=0.1, s=1, label="Non-positive (0)")
    if np.any(pos):
        ax.scatter(means[pos, 0], means[pos, 1], means[pos, 2], c="red", alpha=0.8, s=4, label=f"{title} (1)")
    ax.set_title(f"Label Map: {title}")
    ax.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error(distance, m_i, f_i, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(distance[~f_i], bins=50, alpha=0.5, label="Non-failure (f_i=0)", color="blue", density=True)
    if np.any(f_i):
        ax.hist(distance[f_i], bins=50, alpha=0.6, label="Geometric Failure (f_i=1)", color="red", density=True)
    ax.set_xlabel("Distance $d_i$ to reference surface")
    ax.set_ylabel("Density")
    ax.set_title("Geometric Error Distribution")
    ax.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
