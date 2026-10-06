"""Benchmark label tests.                                         Owner: Trường Thịnh

TODO(TruongThinh): remove the skip marks as functions get implemented.
"""
import pytest


@pytest.mark.skip(reason="TODO(TruongThinh)")
def test_distance_on_sphere():
    """Centers placed at known radii around a sampled sphere -> |d_i - (r_i - R)| < spacing."""


@pytest.mark.skip(reason="TODO(TruongThinh)")
def test_condition_label_ignores_model():
    """m_i must not change when the Gaussian model is perturbed but cameras/reference are fixed
    (only through which reference point is nearest)."""


@pytest.mark.skip(reason="TODO(TruongThinh)")
def test_outcome_label_thresholds():
    """f_i is monotone in tau: increasing tau never adds positives."""


@pytest.mark.skip(reason="TODO(TruongThinh)")
def test_labels_file_roundtrip(tmp_path):
    """save_labels -> np.load gives the keys/dtypes of docs/data_format.md §4."""
