"""D1 metric tests.                                               Owner: Tuấn Khải

TODO(TuanKhai): remove the skip marks as metrics get implemented.
"""
import pytest


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_auroc_perfect_and_inverted():
    """score = label -> 1.0; score = -label -> 0.0; constant score -> 0.5 (ties)."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_against_sklearn_scipy():
    """Random data: auroc/auprc match sklearn, spearman matches scipy (atol 1e-6)."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_precision_and_enrichment_at_k():
    """Hand-computed case: N=100, 10 positives, top-5% contains 3 -> precision 0.6, enrichment 6.0."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_degenerate_labels_return_nan():
    """All-False or all-True labels -> NaN, no exception."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_random_baseline_unbiased():
    """Mean random AUROC ~ 0.5 and mean precision@K ~ prevalence over many draws."""


@pytest.mark.skip(reason="TODO(TuanKhai)")
def test_validate_rejects_mismatched_n(tmp_path):
    """scores.npz and geometric_labels.npz with different N -> validate() raises."""
