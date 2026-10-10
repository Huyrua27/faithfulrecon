"""D1 (failure association) metrics.                            Owner: Tuấn Khải

All functions take score (N,) with HIGHER = LESS reliable and a boolean label (N,).
They must handle ties deterministically (average ranks) and degenerate labels (all 0 / all 1
-> return NaN and let the caller warn).

TODO(TuanKhai):
  - [x] auroc            (reuse measure.auroc; test against sklearn.metrics.roc_auc_score)
  - [x] auprc            (average precision; test against sklearn.metrics.average_precision_score)
  - [x] spearman         score vs continuous geometric error d_i (scipy.stats.spearmanr)
  - [x] precision_at_k   fraction of label=True among the top-k fraction (k in {0.01, 0.05, 0.10})
  - [x] enrichment_at_k  precision_at_k / prevalence  (1.0 = no better than random)
  - [x] tests/test_d1_metrics.py: small hand-computed cases + sklearn/scipy agreement
"""
from __future__ import annotations

import numpy as np

K_FRACTIONS = (0.01, 0.05, 0.10)


def _validate_score_label(score: np.ndarray, label: np.ndarray):
    if score.ndim != 1 or label.ndim != 1:
        raise ValueError("Inputs must be 1-D")

    if len(score) != len(label) or len(score) == 0:
        raise ValueError("Invalid input lengths")

    if label.dtype != np.bool_:
        raise ValueError("label must be boolean")

    if not np.issubdtype(score.dtype, np.floating):
        raise ValueError("score must be float")

    if not np.isfinite(score).all():
        raise ValueError("score must be finite")


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)

    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    ranks = rank_sums[inverse] / counts[inverse]
    return ranks


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    _validate_score_label(score, label)
    pos = int(label.sum())
    neg = int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    ranks = average_ranks(score)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(score: np.ndarray, label: np.ndarray) -> float:
    _validate_score_label(score, label)
    order = np.argsort(score)[::-1]
    sorted_score = score[order]
    sorted_label = label[order]
    n_pos = np.count_nonzero(label)
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    tp_cum = np.cumsum(sorted_label)
    fp_cum = np.cumsum(~sorted_label)
    indices = []
    for i in range(1, len(sorted_score)):
        if sorted_score[i - 1] != sorted_score[i]:
            indices.append(i - 1)
    indices.append(len(sorted_score) - 1)
    precision = tp_cum[indices] / (tp_cum[indices] + fp_cum[indices])
    recall = tp_cum[indices] / n_pos
    delta_recall = []
    prev = 0.0
    for i in range(len(recall)):
        delta_recall.append(recall[i] - prev)
        prev = recall[i]
    average_precision = float((delta_recall * precision).sum())
    return average_precision


def spearman(score: np.ndarray, error: np.ndarray) -> float:
    if score.ndim != 1 or error.ndim != 1:
        raise ValueError("score and error must be 1-D")
    if len(score) != len(error):
        raise ValueError("score and error must have the same length")
    if not (
        np.issubdtype(score.dtype, np.floating)
        and np.issubdtype(error.dtype, np.floating)
    ):
        raise ValueError("score and error must be float")
    if not (np.isfinite(score).all() and np.isfinite(error).all()):
        raise ValueError("score and error must contain only finite values")
    if len(score) == 0:
        raise ValueError("array must not be empty")
    rank_score = average_ranks(score)
    rank_error = average_ranks(error)

    x = rank_score - rank_score.mean()
    y = rank_error - rank_error.mean()

    x_square = np.sum(x**2)
    y_square = np.sum(y**2)
    if x_square == 0 or y_square == 0:
        return float("nan")
    numerator = (x * y).sum()
    denominator = np.sqrt(x_square * y_square)

    return float(numerator / denominator)


def precision_at_k(score: np.ndarray, label: np.ndarray, k_frac: float) -> float:
    _validate_score_label(score, label)
    if not (0 < k_frac <= 1):
        raise ValueError("k_frac must be in (0, 1]")
    n_pos = int(label.sum())
    if n_pos == 0 or n_pos == len(label):
        return float("nan")
    k = max(1, round(k_frac * len(score)))

    cutoff = np.sort(score)[-k]
    above = score > cutoff
    tied = score == cutoff
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


def all_metrics(
    score: np.ndarray, label: np.ndarray, error: np.ndarray | None = None
) -> dict:
    """One row of metrics.csv (without scene/condition/seed/signal columns)."""
    _validate_score_label(score, label)
    spearman_score = spearman(score, error) if error is not None else float("nan")

    result = {
        "prevalence": float(label.mean()),
        "auroc": auroc(score, label),
        "auprc": auprc(score, label),
        "spearman": spearman_score,
    }

    for k_frac in K_FRACTIONS:
        k_percent = round(k_frac * 100)

        result[f"precision_at_{k_percent}"] = precision_at_k(score, label, k_frac)
        result[f"enrichment_at_{k_percent}"] = enrichment_at_k(
            score, label, k_frac
        )

    return result
