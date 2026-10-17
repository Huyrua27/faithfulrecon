"""Synthetic regression checks for D1 failure-association metrics."""

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from frecon.eval import d1_metrics as metrics


@pytest.fixture
def tied_case():
    score = np.array([0.9, 0.9, 0.2, 0.1], dtype=float)
    label = np.array([True, False, True, False], dtype=bool)
    error = np.array([0.1, 0.8, 0.4, 0.2], dtype=float)
    return score, label, error


def test_auroc_perfect_inverted_and_constant_scores():
    label = np.array([True, False, True, False], dtype=bool)
    score = label.astype(float)

    assert metrics.auroc(score, label) == pytest.approx(1.0)
    assert metrics.auroc(-score, label) == pytest.approx(0.0)
    assert metrics.auroc(np.full(len(label), 0.5), label) == pytest.approx(0.5)


def test_auroc_hand_computed_pairs():
    score = np.array([0.9, 0.7, 0.6, 0.2])
    label = np.array([True, False, True, False])
    # The two positives win 2 and 1 of their 2 comparisons, respectively.
    assert metrics.auroc(score, label) == pytest.approx(3 / 4)


def test_average_precision_hand_computed():
    score = np.array([0.9, 0.8, 0.7, 0.1])
    label = np.array([False, True, False, True])
    # Precision is 1/2 at the first positive and 2/4 at the second.
    assert metrics.auprc(score, label) == pytest.approx((1 / 2 + 2 / 4) / 2)


def test_auroc_matches_sklearn_with_ties(tied_case):
    score, label, _ = tied_case
    assert metrics.auroc(score, label) == pytest.approx(roc_auc_score(label, score))


def test_average_precision_matches_sklearn_with_ties(tied_case):
    score, label, _ = tied_case
    assert metrics.auprc(score, label) == pytest.approx(
        average_precision_score(label, score)
    )


@pytest.mark.parametrize("n,n_levels", [(2, 2), (17, 1), (101, 7), (1000, 12)])
def test_average_precision_matches_sklearn_for_changed_tie_patterns(n, n_levels):
    rng = np.random.default_rng(20261017)
    score = rng.integers(0, n_levels, size=n).astype(float) - n_levels // 2
    label = rng.random(n) < 0.2
    label[0], label[-1] = True, False

    assert metrics.auprc(score, label) == pytest.approx(
        average_precision_score(label, score), abs=1e-12
    )


def test_spearman_matches_scipy(tied_case):
    score, _, error = tied_case
    assert metrics.spearman(score, error) == pytest.approx(
        spearmanr(score, error).statistic
    )


def test_average_ranks_match_scipy(tied_case):
    score, _, _ = tied_case
    np.testing.assert_allclose(
        metrics.average_ranks(score), rankdata(score, method="average")
    )


def test_precision_and_enrichment_at_k():
    label = np.array([True, False, True, False, False, False], dtype=bool)
    score_a = np.array([0.2, 0.9, 0.01, 0.99, 0.55, 1.0])
    score_b = np.array([1.2, 0.9, 0.01, 0.99, 0.55, 1.0])

    assert metrics.precision_at_k(score_a, label, 0.25) == pytest.approx(0.0)
    assert metrics.enrichment_at_k(score_a, label, 0.25) == pytest.approx(0.0)
    assert metrics.precision_at_k(score_b, label, 0.25) == pytest.approx(0.5)
    assert metrics.enrichment_at_k(score_b, label, 0.25) == pytest.approx(1.5)


def test_precision_and_enrichment_at_tied_cutoff(tied_case):
    score, label, _ = tied_case
    # One top-K slot is shared equally by a positive and a negative.
    assert metrics.precision_at_k(score, label, 0.25) == pytest.approx(0.5)
    assert metrics.enrichment_at_k(score, label, 0.25) == pytest.approx(1.0)


def test_all_metrics_has_ten_keys(tied_case):
    score, label, error = tied_case
    expected = {"prevalence", "auroc", "auprc", "spearman"}
    for k in (1, 5, 10):
        expected.update({f"precision_at_{k}", f"enrichment_at_{k}"})

    result = metrics.all_metrics(score, label, error)
    assert set(result) == expected
    assert result["prevalence"] == pytest.approx(0.5)


def test_all_metrics_without_error_has_nan_spearman(tied_case):
    score, label, _ = tied_case
    assert np.isnan(metrics.all_metrics(score, label)["spearman"])


@pytest.mark.parametrize("label_value", [False, True])
def test_degenerate_labels_return_nan(tied_case, label_value):
    score, _, _ = tied_case
    label = np.full(len(score), label_value, dtype=bool)

    assert np.isnan(metrics.auroc(score, label))
    assert np.isnan(metrics.auprc(score, label))
    assert np.isnan(metrics.precision_at_k(score, label, 0.25))
    assert np.isnan(metrics.enrichment_at_k(score, label, 0.25))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        metrics.all_metrics(np.array([], dtype=float), np.array([], dtype=bool))


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_nonfinite_score_is_rejected(bad_score):
    with pytest.raises(ValueError):
        metrics.all_metrics(np.array([bad_score, 0.2]), np.array([True, False]))


def test_nonboolean_label_is_rejected(tied_case):
    score, _, _ = tied_case
    with pytest.raises(ValueError):
        metrics.auroc(score, np.array([1, 0, 1, 0]))


@pytest.mark.parametrize("k_frac", [-0.1, 0.0, 1.2])
def test_invalid_k_fraction_is_rejected(tied_case, k_frac):
    score, label, _ = tied_case
    with pytest.raises(ValueError):
        metrics.precision_at_k(score, label, k_frac)


def test_mismatched_score_and_label_lengths_are_rejected(tied_case):
    score, label, _ = tied_case
    with pytest.raises(ValueError):
        metrics.auroc(score, label[:-1])


def test_mismatched_error_length_is_rejected(tied_case):
    score, label, _ = tied_case
    with pytest.raises(ValueError):
        metrics.all_metrics(score, label, np.array([0.1, 0.2]))
