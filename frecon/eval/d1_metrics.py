"""D1 (failure association) metrics.                            Owner: Tuấn Khải

All functions take score (N,) with HIGHER = LESS reliable and a boolean label (N,).
They must handle ties deterministically (average ranks) and degenerate labels (all 0 / all 1
-> return NaN and let the caller warn).

Existing code: frecon/measure.py::auroc() (Mann–Whitney with tie correction) — move it here.

TODO(TuanKhai):
  - [ ] auroc            (reuse measure.auroc; test against sklearn.metrics.roc_auc_score)
  - [ ] auprc            (average precision; test against sklearn.metrics.average_precision_score)
  - [ ] spearman         score vs continuous geometric error d_i (scipy.stats.spearmanr)
  - [ ] precision_at_k   fraction of label=True among the top-k fraction (k in {0.01, 0.05, 0.10})
  - [ ] enrichment_at_k  precision_at_k / prevalence  (1.0 = no better than random)
  - [ ] tests/test_d1_metrics.py: small hand-computed cases + sklearn/scipy agreement
"""
from __future__ import annotations

import numpy as np

K_FRACTIONS = (0.01, 0.05, 0.10)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    raise NotImplementedError("TODO(TuanKhai)")


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    raise NotImplementedError("TODO(TuanKhai)")


def spearman(score: np.ndarray, error: np.ndarray) -> float:
    raise NotImplementedError("TODO(TuanKhai)")


def precision_at_k(score: np.ndarray, label: np.ndarray, k_frac: float) -> float:
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    k = max(1, round(k_frac * len(score)))
    id = np.argsort(score)[-k:]
    sum_f_i = label[id].sum()
    return float(sum_f_i) / float(k)


def enrichment_at_k(score: np.ndarray, label: np.ndarray, k_frac: float) -> float:
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")

    precision_k = precision_at_k(score, label, k_frac)
    f_i_true = label.sum() / len(score)
    enrichment_k = precision_k / f_i_true
    return enrichment_k


def all_metrics(score: np.ndarray, label: np.ndarray, error: np.ndarray | None = None) -> dict:
    """One row of metrics.csv (without scene/condition/seed/signal columns)."""
    raise NotImplementedError("TODO(TuanKhai)")
