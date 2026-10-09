"""Second-order signal tests.                                     Owner: Phát

TODO(Phat): remove the skip marks as functions get implemented.
"""
import pytest


@pytest.mark.skip(reason="TODO(Phat)")
def test_fisher_blocks_match_exact_jacobian():
    """Tiny scene (~20 Gaussians, 16x16 image): exact J via torch.autograd.functional.jacobian,
    F_exact = J^T J per Gaussian block; estimate with n_samples = 4, 64, 1024 -> relative error
    decreases roughly like 1/sqrt(n_samples)."""


@pytest.mark.skip(reason="TODO(Phat)")
@pytest.mark.parametrize("name", ["fisher", "pup", "curvature", "s_geom", "s_geom_ranksum"])
def test_contract(name):
    """compute_signals(..., names=(name,)) -> shape (N,), float32, finite."""


@pytest.mark.skip(reason="TODO(Phat)")
def test_curvature_orientation():
    """A Gaussian seen by more views (larger F_i) gets a lower curvature score u_i."""
