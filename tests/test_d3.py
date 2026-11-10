"""D3 tests.                                                      Owner: Phát

TODO(Phat): remove the skip marks as functions get implemented.
"""
import pytest


@pytest.mark.skip(reason="TODO(Phat)")
def test_attribution_trick_matches_explicit_weights():
    """On a tiny scene, sum_p w_{i,p} err_p from the feature-gradient trick equals the value
    computed from explicit blending weights (dense reference in tests/test_rasterizer.py)."""


@pytest.mark.skip(reason="TODO(Phat)")
def test_ause_oracle_is_zero():
    """pred_map == err_map -> AUSE == 0; random pred_map -> AUSE > 0."""
