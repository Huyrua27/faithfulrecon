"""Synthetic checks for the matched random D1 ranking control."""

import numpy as np
import pytest

from frecon.eval.random_baseline import random_reference, random_scores


def test_random_scores_are_reproducible_and_change_between_draws():
    first = random_scores(20, seed=7, draw=3)
    np.testing.assert_array_equal(first, random_scores(20, seed=7, draw=3))
    assert first.shape == (20,)
    assert not np.array_equal(first, random_scores(20, seed=7, draw=4))


def test_random_reference_is_reproducible_and_unbiased_on_synthetic_labels():
    label = np.array([True] * 4 + [False] * 16, dtype=bool)
    error = np.arange(20, dtype=float)
    summary = random_reference(label, error, R=1000, seed=7)

    assert summary == random_reference(label, error, R=1000, seed=7)
    assert summary["prevalence"][0] == pytest.approx(0.2)
    assert summary["auroc"][0] == pytest.approx(0.5, abs=0.04)
    assert summary["precision_at_10"][0] == pytest.approx(0.2, abs=0.04)
    assert summary["spearman"][0] == pytest.approx(0.0, abs=0.04)
    for mean, std, q05, q95 in summary.values():
        assert std >= 0
        assert q05 <= q95


@pytest.mark.parametrize("label_value", [False, True])
def test_undefined_metrics_stay_nan_instead_of_becoming_zero(label_value):
    label = np.full(4, label_value, dtype=bool)
    summary = random_reference(label, None, R=3, seed=7)

    assert summary["prevalence"][0] == float(label_value)
    assert all(np.isnan(value) for value in summary["auroc"])
    assert all(np.isnan(value) for value in summary["spearman"])


def test_constant_error_keeps_spearman_undefined():
    label = np.array([True, False, True, False], dtype=bool)
    error = np.ones(4, dtype=float)
    summary = random_reference(label, error, R=3, seed=7)

    assert all(np.isnan(value) for value in summary["spearman"])
    assert np.isfinite(summary["auroc"][0])


def test_nonpositive_draw_count_and_size_are_rejected():
    with pytest.raises(ValueError):
        random_scores(0, seed=7, draw=0)
    with pytest.raises(ValueError):
        random_reference(np.array([True, False]), None, R=0)
