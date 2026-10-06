"""Signal contract tests (shape, orientation, finiteness).        Owner: Quỳnh Anh

Run: pytest tests/test_signals.py      (CPU is enough; uses a tiny random scene)
"""
import pytest
import torch

from frecon.camera import orbit_cameras
from frecon.gaussians import GaussianModel
from frecon.render import get_rasterizer
from frecon.signals import REGISTRY, compute_signals, top_fraction


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = GaussianModel.random_in_box(300, 0.8, g, "cpu")
    cams = orbit_cameras(4, 4.0, 32, 32, 30.0)
    rast = get_rasterizer("torch")
    bg = torch.ones(3)
    for c in cams:
        c.image = torch.rand(32, 32, 3)
    return model, cams, rast, bg


@pytest.mark.parametrize("name", ["random", "opacity", "gradmag", "visibility"])
def test_contract(tiny, name):
    model, cams, rast, bg = tiny
    s = compute_signals(model, cams, rast, bg, names=(name,))[name]
    assert s.shape == (model.n,)
    assert s.dtype == torch.float32
    assert torch.isfinite(s).all()


def test_opacity_orientation(tiny):
    """Lower opacity must give a higher (less reliable) score."""
    model, cams, rast, bg = tiny
    s = compute_signals(model, cams, rast, bg, names=("opacity",))["opacity"]
    op = model.opacities.detach()
    assert s[op.argmin()] == s.max()


def test_top_fraction():
    s = torch.arange(100, dtype=torch.float32)
    m = top_fraction(s, 0.05)
    assert m.sum() == 5 and m[95:].all()


@pytest.mark.skip(reason="TODO(QuynhAnh): implement gradient_inconsistency first")
def test_gradient_inconsistency(tiny):
    # TODO(QuynhAnh): contract + a constructed case where two views give opposite gradients
    model, cams, rast, bg = tiny
    s = REGISTRY["gradient_inconsistency"]
    raise NotImplementedError
