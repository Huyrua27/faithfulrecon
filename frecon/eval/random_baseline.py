"""Matched random baseline for D1.                              Owner: Tuấn Khải

Matched = same scene, same condition, same K, same seed (same Gaussians, same labels);
only the ranking is random. Report the mean and a spread over R random draws, so a signal's
metric can be compared with what random selection gives on exactly the same data.

TODO(TuanKhai):
  - [x] random_scores(n, seed, draw) -> (N,) deterministic for (seed, draw).
  - [x] random_reference(labels, error, R=100, seed) -> dict metric -> (mean, std, q05, q95).
  - [x] Note: for AUROC the expectation is 0.5 and for precision@K it is the prevalence —
        use these as a check that the implementation is unbiased.
"""

from __future__ import annotations

import numpy as np

from frecon.eval.d1_metrics import all_metrics


def random_scores(n: int, seed: int, draw: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    return rng.random(n)


def random_reference(
    label: np.ndarray, error: np.ndarray | None, R: int = 100, seed: int = 0
) -> dict:
    if R <= 0:
        raise ValueError("R must be positive")

    results = []
    for draw in range(R):
        random_score = random_scores(n=len(label), seed=seed, draw=draw)
        results.append(all_metrics(random_score, label, error))

    summary = {}
    for metric_name in results[0].keys():
        values = np.array([result[metric_name] for result in results], dtype=float)
        summary[metric_name] = (
            float(values.mean()),
            float(values.std()),
            float(np.quantile(values, 0.05)),
            float(np.quantile(values, 0.95)),
        )
    return summary
