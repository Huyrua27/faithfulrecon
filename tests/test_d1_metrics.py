"""D1 metric tests.                                               Owner: Tuấn Khải

TODO(TuanKhai): remove the skip marks as metrics get implemented.
"""
import pytest
import numpy as np
from frecon.eval.d1_metrics import precision_at_k, enrichment_at_k


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_auroc_perfect_and_inverted():
    """score = label -> 1.0; score = -label -> 0.0; constant score -> 0.5 (ties)."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_against_sklearn_scipy():
    """Random data: auroc/auprc match sklearn, spearman matches scipy (atol 1e-6)."""
def test_precision_and_enrichment_at_k():
    label = np.array([True, False, True, False, False, False], dtype=bool)
    score_a = np.array([0.2, 0.9, 0.01, 0.99, 0.55, 1.0])
    score_b = np.array([1.2, 0.9, 0.01, 0.99, 0.55, 1.0])

    assert precision_at_k(score_a, label, 0.25) == 0.0
    assert enrichment_at_k(score_a, label, 0.25) == 0.0
    assert precision_at_k(score_b, label, 0.25) == 0.5
    assert enrichment_at_k(score_b, label, 0.25) == 1.5


def test_degenerate_labels_return_nan():
    score = np.array([0.9, 0.2, 0.5, 0.1])
    for label in (
        np.zeros(4, dtype=bool),
        np.ones(4, dtype=bool),
    ):
        assert np.isnan(precision_at_k(score, label, 0.5))
        assert np.isnan(enrichment_at_k(score, label, 0.5))


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_random_baseline_unbiased():
    """Mean random AUROC ~ 0.5 and mean precision@K ~ prevalence over many draws."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_validate_rejects_mismatched_n(tmp_path):
    """scores.npz and geometric_labels.npz with different N -> validate() raises."""
