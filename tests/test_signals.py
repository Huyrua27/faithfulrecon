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


@pytest.mark.parametrize("name", ["random", "opacity", "gradmag", "visibility", "gradient_inconsistency"])
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


def test_gradient_inconsistency(tiny):
    model, cams, rast, bg = tiny
    s = compute_signals(model, cams, rast, bg, names=("gradient_inconsistency",))
    assert (s["gradient_inconsistency"] >= 0).all()
    assert s["_gi_count"].shape == (model.n,)


def controlled_context(monkeypatch, gradients, visible=None):
    """Linear per-view losses with known world-position gradients (no raster approximation)."""
    from types import SimpleNamespace
    from frecon.signals import gradient_inconsistency as gi, SignalContext
    gradients = torch.as_tensor(gradients, dtype=torch.float32)
    n = gradients.shape[1]
    means = torch.nn.Parameter(torch.zeros(n, 3))
    if visible is None:
        visible = torch.ones(gradients.shape[:2])

    class LinearModel:
        def __init__(self):
            self.n = n
            self.means = means

        def render(self, rasterize, cam, bg, **kw):
            return {"image": (self.means * cam.g).sum(), "contrib": cam.contrib}

    monkeypatch.setattr(gi, "photometric_loss", lambda image, target, weight: image)
    cams = [SimpleNamespace(g=g, contrib=torch.as_tensor(v), image=None)
            for g, v in zip(gradients, visible)]
    return SignalContext(LinearModel(), cams, None, torch.ones(3))


def test_identical_opposite_zero_and_missing_gradients(monkeypatch):
    from frecon.signals.gradient_inconsistency import compute
    # i=0 identical; i=1 opposite; i=2 all zero; i=3 one view; i=4 unseen.
    g = [[[1, 0, 0], [1, 0, 0], [0, 0, 0], [5, 0, 0], [4, 0, 0]],
         [[1, 0, 0], [-1, 0, 0], [0, 0, 0], [9, 0, 0], [8, 0, 0]]]
    ctx = controlled_context(monkeypatch, g, [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    score = compute(ctx, delta=0.01)
    torch.testing.assert_close(score, torch.tensor([0., 100., 0., 0., 0.]))
    assert ctx.cache["gradient_inconsistency_stats"]["count"].tolist() == [2, 2, 2, 1, 0]


def test_streaming_matches_explicit_covariance_and_rotation(monkeypatch):
    from frecon.signals.gradient_inconsistency import compute
    g = torch.randn(7, 8, 3, generator=torch.Generator().manual_seed(4))
    ctx = controlled_context(monkeypatch, g)
    actual = compute(ctx, delta=0.01)
    mean = g.double().mean(0)
    variance = (g.double() - mean).square().sum(-1).mean(0)
    expected = (variance / (mean.square().sum(-1) + 0.01)).float()
    torch.testing.assert_close(actual, expected)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotated = controlled_context(monkeypatch, g @ rotation)
    torch.testing.assert_close(compute(rotated, delta=0.01), actual, rtol=2e-5, atol=1e-5)


def test_gradient_amplitude_variation_is_not_only_direction(monkeypatch):
    from frecon.signals.gradient_inconsistency import compute
    ctx = controlled_context(monkeypatch, [[[1, 0, 0]], [[3, 0, 0]]])
    torch.testing.assert_close(compute(ctx, delta=1.), torch.tensor([0.2]))


def test_inconsistency_cache_respects_visibility_threshold(monkeypatch):
    from frecon.signals.gradient_inconsistency import compute
    ctx = controlled_context(monkeypatch, [[[1, 0, 0]], [[-1, 0, 0]]], [[0.1], [0.01]])
    assert compute(ctx, delta=0.01, contrib_eps=0.001).item() == 100
    assert compute(ctx, delta=0.01, contrib_eps=0.05).item() == 0


def test_relative_delta_and_joint_gradient_delta_scaling(monkeypatch):
    from frecon.signals.gradient_inconsistency import compute
    gradients = torch.tensor([[[1., 0, 0]], [[3., 0, 0]]])
    ctx = controlled_context(monkeypatch, gradients)
    score = compute(ctx, delta=1., delta_rel=.5)
    assert ctx.cache["gradient_inconsistency_stats"]["effective_delta"] == 3
    torch.testing.assert_close(score, torch.tensor([1 / 7]))
    scaled = controlled_context(monkeypatch, gradients * 10)
    torch.testing.assert_close(compute(scaled, delta=100., delta_rel=.5), score)


@pytest.mark.parametrize("kwargs", [{"delta": 0}, {"delta": float("nan")},
    {"delta_rel": -1}, {"n_min": 1}, {"contrib_eps": -1}, {"lambda_dssim": 2}])
def test_inconsistency_invalid_parameters(tiny, kwargs):
    from frecon.signals import SignalContext
    with pytest.raises(ValueError):
        REGISTRY["gradient_inconsistency"](SignalContext(*tiny), **kwargs)


def test_scoring_preserves_parameters_and_existing_grad_buffers(tiny):
    model, cams, rast, bg = tiny
    before = model.state_dict()
    for p in model.params.values():
        p.grad = torch.ones_like(p)
    try:
        compute_signals(model, cams, rast, bg, names=("gradmag", "gradient_inconsistency"))
        for name, p in model.params.items():
            torch.testing.assert_close(p, before[name])
            torch.testing.assert_close(p.grad, torch.ones_like(p))
    finally:
        for p in model.params.values():
            p.grad = None


def test_gradmag_matches_legacy_backward_and_parameterized_cache(tiny):
    from frecon.signals import SignalContext, view_statistics
    from frecon.losses import photometric_loss
    model, cams, rast, bg = tiny
    ctx = SignalContext(*tiny)
    actual = view_statistics(ctx)
    total, count = torch.zeros(model.n), torch.zeros(model.n)
    for cam in cams:
        out = model.render(rast, cam, bg, return_contrib=True)
        out["means2d"].retain_grad()
        photometric_loss(out["image"], cam.image, 0.2).backward()
        g = out["means2d"].grad
        norm = torch.stack([g[:, 0] * cam.W / 2, g[:, 1] * cam.H / 2], -1).norm(dim=-1)
        seen = out["contrib"] > 1e-3
        total[seen] += norm[seen]
        count[seen] += 1
    for p in model.params.values():
        p.grad = None
    torch.testing.assert_close(actual["grad_mean"], total / count.clamp_min(1))
    torch.testing.assert_close(actual["vis_count"], count)
    empty = view_statistics(ctx, contrib_eps=1e10)
    assert empty["vis_count"].sum() == 0
    assert view_statistics(ctx)["vis_count"].sum() == count.sum()
