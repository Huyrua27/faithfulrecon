"""D1 (failure association) metrics.                            Owner: Tuấn Khải

All functions take score (N,) with HIGHER = LESS reliable and a boolean label (N,).
They must handle ties deterministically (average ranks) and degenerate labels (all 0 / all 1
-> return NaN and let the caller warn).

TODO(TuanKhai):
  - [x] auroc            (reuse measure.auroc; test against sklearn.metrics.roc_auc_score)
  - [x] auprc            (average precision; test against sklearn.metrics.average_precision_score)
  - [ ] spearman         score vs continuous geometric error d_i (scipy.stats.spearmanr)
  - [x] precision_at_k   fraction of label=True among the top-k fraction (k in {0.01, 0.05, 0.10})
  - [x] enrichment_at_k  precision_at_k / prevalence  (1.0 = no better than random)
  - [ ] tests/test_d1_metrics.py: small hand-computed cases + sklearn/scipy agreement
"""
from __future__ import annotations

import numpy as np

K_FRACTIONS = (0.01, 0.05, 0.10)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    if score.ndim != 1 or label.ndim != 1:
        raise ValueError("score and label must be 1-D")
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    if label.dtype != np.bool_:
        raise ValueError("label must be boolean")
    if not np.isfinite(score).all():
        raise ValueError("score must contain only finite values")

    pos = int(label.sum())
    neg = int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score),dtype=float)
    ranks[order] = np.arange(1,len(score)+1,dtype=float)
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    ranks = rank_sums[inverse] / counts[inverse]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))

def auprc(score: np.ndarray, label: np.ndarray) -> float:
    if score.ndim != 1 or label.ndim != 1:
        raise ValueError("score and label must be 1-D")
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    if label.dtype != np.bool_:
        raise ValueError("label must be boolean")
    if not np.isfinite(score).all():
        raise ValueError("score must contain only finite values")
    if len(score) == 0:
        raise ValueError("array must not be empty")
    order = np.argsort(score)[::-1]
    sorted_score = score[order]
    sorted_label = label[order]
    n_pos = np.count_nonzero(label)
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    tp_cum = np.cumsum(sorted_label)
    fp_cum = np.cumsum(~sorted_label)
    indices = []
    for i in range(1,len(sorted_score)):
        if sorted_score[i-1] != sorted_score[i]:
            indices.append(i-1)
    indices.append(len(sorted_score)-1)
    precision = tp_cum[indices]/(tp_cum[indices]+fp_cum[indices])
    recall = tp_cum[indices]/n_pos
    delta_recall = []
    prev = 0.0
    for i in range(len(recall)):
        delta_recall.append(recall[i] - prev)
        prev = recall[i]
    average_precision = float((delta_recall*precision).sum())
    return average_precision

def spearman(score: np.ndarray, error: np.ndarray) -> float:
    raise NotImplementedError("TODO(TuanKhai)")


def precision_at_k(score: np.ndarray, label: np.ndarray, k_frac: float) -> float:

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    if not (0 < k_frac <= 1):
        raise ValueError("k_frac must be in (0, 1]")
    if not np.isfinite(score).all():
        raise ValueError("score must contain only finite values")
    if score.ndim != 1 or label.ndim != 1:
        raise ValueError("score and label must be 1-D")
    if label.dtype != np.bool_:
        raise ValueError("label must be boolean")

    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    k = max(1, round(k_frac * len(score)))

    cutoff = np.sort(score)[-k]
    above = (score>cutoff)
    tied = (score==cutoff)
    slots = k - int(above.sum())
    expected_true = label[above].sum() + (slots / tied.sum()) * label[tied].sum()
    return float(expected_true) / float(k)


def enrichment_at_k(score: np.ndarray, label: np.ndarray, k_frac: float) -> float:
    precision_k = precision_at_k(score, label, k_frac)
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")

    f_i_true = label.sum() / len(score)
    enrichment_k = precision_k / f_i_true
    return enrichment_k


def all_metrics(score: np.ndarray, label: np.ndarray, error: np.ndarray | None = None) -> dict:
    """One row of metrics.csv (without scene/condition/seed/signal columns)."""
    raise NotImplementedError("TODO(TuanKhai)")
