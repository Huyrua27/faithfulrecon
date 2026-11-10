"""D3 — held-out predictive validity.                            Owner: Phát

Spec: docs/tasks/Phat_tasks.md §2.

Per-Gaussian held-out error without touching the renderer:
    render with features = ones(N,1) (requires_grad), compute sum_p err_p * out_p, backward
    -> grad of feature i = sum_p w_{i,p} err_p ;  denominator = contrib (return_contrib=True)
    e_i = sum over held-out views of numerator / sum of denominator

Pixel-level: render the score as a 1-channel feature map -> per-pixel predicted unreliability
-> sparsification curve vs the oracle (pixels sorted by true error) -> AUSE.

TODO(Phat):
  - [ ] heldout_error_per_gaussian(model, test_cams, rasterize, bg, err="l1"|"dssim")
  - [ ] score_maps(model, score, test_cams, ...) -> (V,H,W); normalize score to [0,1] first
        (rendering mixes scores by blending weights — document this choice)
  - [ ] sparsification(err_map, pred_map, fractions) and ause(...)
  - [ ] d3_metrics(score, e_i, f_i, err_maps, pred_maps) -> dict (spearman_e, auroc_f, ause)
  - [ ] matched random baseline (same as D1: frecon/eval/random_baseline.py when available)
  - [ ] tests/test_d3.py: attribution trick equals explicit per-pixel weights on a tiny scene
"""
from __future__ import annotations

import torch


def heldout_error_per_gaussian(model, test_cams, rasterize, bg, err: str = "l1") -> torch.Tensor:
    raise NotImplementedError("TODO(Phat)")


def score_maps(model, score, test_cams, rasterize, bg) -> torch.Tensor:
    raise NotImplementedError("TODO(Phat)")


def sparsification(err_map, pred_map, fractions=None):
    raise NotImplementedError("TODO(Phat)")


def ause(err_map, pred_map) -> float:
    raise NotImplementedError("TODO(Phat)")


def d3_metrics(score, e_i, f_i, err_maps, pred_maps) -> dict:
    raise NotImplementedError("TODO(Phat)")
