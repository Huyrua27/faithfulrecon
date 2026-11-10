"""Renderer backends. All backends share the signature of `torch_rast.rasterize`.

- "torch": pure PyTorch (default; runs on any GPU/CPU, slow).
- "gsplat": CUDA rasterizer from the gsplat package (fast; needs a gsplat build that
  matches your torch/CUDA/Python). Untested in this repo; verify against "torch" on a
  few views before using it for experiments.
"""
from __future__ import annotations

from . import torch_rast


def get_rasterizer(backend: str = "torch"):
    if backend == "torch":
        return torch_rast.rasterize
    if backend == "gsplat":
        from .gsplat_rast import rasterize
        return rasterize
    raise ValueError(f"unknown renderer backend: {backend}")
