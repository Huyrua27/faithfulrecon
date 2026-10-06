"""Matched random baseline for D1.                              Owner: Tuấn Khải

Matched = same scene, same condition, same K, same seed (same Gaussians, same labels);
only the ranking is random. Report the mean and a spread over R random draws, so a signal's
metric can be compared with what random selection gives on exactly the same data.

TODO(TuanKhai):
  - [ ] random_scores(n, seed, draw) -> (N,) deterministic for (seed, draw).
  - [ ] random_reference(labels, error, R=100, seed) -> dict metric -> (mean, std, q05, q95).
  - [ ] Note: for AUROC the expectation is 0.5 and for precision@K it is the prevalence —
        use these as a check that the implementation is unbiased.
"""
from __future__ import annotations

import numpy as np


def random_scores(n: int, seed: int, draw: int) -> np.ndarray:
    raise NotImplementedError("TODO(TuanKhai)")


def random_reference(label: np.ndarray, error: np.ndarray | None, R: int = 100, seed: int = 0) -> dict:
    raise NotImplementedError("TODO(TuanKhai)")
