"""Second-order estimator tests and a reproducible tiny-scene convergence study.

Tests: python -m pytest tests/test_second_order.py
Evidence: python -m tests.test_second_order --out_dir results/second_order
"""
from dataclasses import replace

import pytest
import torch

from frecon.camera import orbit_cameras
from frecon.gaussians import GaussianModel, inverse_sigmoid
from frecon.render import get_rasterizer
from frecon.signals.common import SignalContext
from frecon.signals.second_order import fisher_blocks


def tiny_context(seed=0, device="cpu"):
    """20 Gaussians, two 16x16 training cameras, no GT/checkpoint dependency."""
    generator = torch.Generator().manual_seed(123)
    n = 20
    model = GaussianModel(
        (torch.rand(n, 3, generator=generator) - 0.5) * 1.2,
        torch.log(0.10 + 0.10 * torch.rand(n, 3, generator=generator)),
        torch.randn(n, 4, generator=generator),
        inverse_sigmoid(torch.linspace(0.35, 0.85, n)),
        inverse_sigmoid(0.1 + 0.8 * torch.rand(n, 3, generator=generator)),
    )
    if device != "cpu":
        model = GaussianModel.from_state_dict(model.state_dict(), device)
    cams = [c.to(device) for c in orbit_cameras(2, 4.0, 16, 16, 30.0)]
    return SignalContext(model, cams, get_rasterizer("torch"), torch.ones(3, device=device), seed)


def exact_blocks(ctx, params):
    """Independent oracle: exact functional image Jacobian, then per-row J.T @ J."""
    values = tuple(ctx.model.params[name].detach().clone() for name in params)
    fixed = {name: value.detach() for name, value in ctx.model.params.items()}
    blocks = None
    for cam in ctx.cams:
        def render(*inputs):
            p = dict(fixed, **dict(zip(params, inputs)))
            return ctx.rasterize(
                p["means"], p["quats"], p["log_scales"].exp(),
                p["opacity_logits"].sigmoid(), p["color_logits"].sigmoid(), cam, ctx.bg,
            )["image"].flatten()

        jacobians = torch.autograd.functional.jacobian(render, values, vectorize=False)
        jacobian = torch.cat([j.reshape(j.shape[0], ctx.model.n, -1) for j in jacobians], dim=-1)
        per_view = torch.einsum("pid,pie->ide", jacobian, jacobian)
        blocks = per_view if blocks is None else blocks + per_view
    return blocks


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    """Small CPU splats avoid thread-launch overhead; restore the user's setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("params", [("means",), ("means", "log_scales")])
def test_fisher_blocks_match_exact_jacobian(params):
    ctx = tiny_context(seed=7)
    exact = exact_blocks(ctx, params)
    estimate = fisher_blocks(ctx, params, n_samples=256)
    relative_error = (estimate - exact).norm() / exact.norm()
    # Monte Carlo accuracy on a fixed scene/seed, not exact equality of a random estimate.
    assert relative_error < 0.20, relative_error.item()
    assert estimate.shape == exact.shape
    assert estimate.dtype == torch.float32 and not estimate.requires_grad
    torch.testing.assert_close(estimate, estimate.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(estimate)
    assert eigenvalues.min() >= -1e-6 * estimate.norm()


def linear_rasterize(means, quats, scales, opacities, features, cam, bg):
    """One-output analytical renderer; any Rademacher probe gives its exact block."""
    direction = means.new_tensor([1.0, 2.0, -0.5])
    value = ((means @ direction) * opacities).sum()
    return {"image": value.reshape(1, 1, 1)}


@pytest.mark.parametrize("n_samples", [1, 4, 17])
def test_sample_average_view_sum_and_scalar_parameters(n_samples):
    ctx = tiny_context()
    ctx.rasterize = linear_rasterize
    means, opacity = ctx.model.means.detach(), ctx.model.opacities.detach()
    direction = means.new_tensor([1.0, 2.0, -0.5])
    # Request an unusual order, a scalar field, and an unused field (log_scales).
    gradient = torch.cat([
        ((means @ direction) * opacity * (1 - opacity))[:, None],
        torch.zeros_like(means), opacity[:, None] * direction,
    ], dim=1)
    exact = len(ctx.cams) * gradient[:, :, None] * gradient[:, None, :]
    estimate = fisher_blocks(ctx, ("opacity_logits", "log_scales", "means"), n_samples)
    torch.testing.assert_close(estimate, exact)


def test_probe_order_cache_and_rng_independence():
    ctx = tiny_context(seed=42)
    before = torch.random.get_rng_state().clone()
    means = fisher_blocks(ctx, n_samples=8)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert fisher_blocks(ctx, n_samples=8) is means
    fresh = replace(ctx, cache={})
    torch.rand(100, generator=fresh.generator)  # another signal has consumed the shared RNG
    both = fisher_blocks(fresh, ("means", "log_scales"), n_samples=8)
    torch.testing.assert_close(both[:, :3, :3], means, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(fisher_blocks(fresh, n_samples=8), means, rtol=0, atol=0)
    assert fisher_blocks(ctx, n_samples=9) is not means
    changed_seed = fisher_blocks(replace(ctx, seed=43), n_samples=8)
    assert not torch.equal(changed_seed, means)


def test_model_values_and_existing_gradients_are_preserved():
    ctx = tiny_context()
    values = ctx.model.state_dict()
    for index, p in enumerate(ctx.model.params.values()):
        p.grad = torch.full_like(p, index + 1.0) if index % 2 else None
    grads = {name: p.grad for name, p in ctx.model.params.items()}
    with torch.no_grad():
        blocks = fisher_blocks(ctx, ("means", "log_scales"), n_samples=4)
    assert not blocks.requires_grad
    for index, (name, p) in enumerate(ctx.model.params.items()):
        torch.testing.assert_close(p, values[name], rtol=0, atol=0)
        assert p.grad is grads[name]
        if p.grad is not None:
            torch.testing.assert_close(p.grad, torch.full_like(p, index + 1.0), rtol=0, atol=0)


@pytest.mark.parametrize("all_invisible", [False, True])
def test_invisible_gaussians_have_zero_information(all_invisible):
    ctx = tiny_context()
    with torch.no_grad():
        ctx.model.params["opacity_logits"][-1 if not all_invisible else slice(None)] = -100.0
    blocks = fisher_blocks(ctx, ("means", "log_scales"), n_samples=4)
    assert torch.count_nonzero(blocks[-1]) == 0
    if all_invisible:
        assert torch.count_nonzero(blocks) == 0
    else:
        assert blocks[:-1].norm() > 0


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_invalid_sample_counts(n_samples):
    with pytest.raises(ValueError, match="positive integer"):
        fisher_blocks(tiny_context(), n_samples=n_samples)


@pytest.mark.parametrize("params", [(), ("means", "means"), ("missing",), "means"])
def test_invalid_parameters(params):
    with pytest.raises(ValueError, match="parameter"):
        fisher_blocks(tiny_context(), params)


def test_empty_views_and_frozen_parameters():
    ctx = tiny_context()
    with pytest.raises(ValueError, match="training camera"):
        fisher_blocks(replace(ctx, cams=[]))
    ctx.model.params["means"].requires_grad_(False)
    with pytest.raises(ValueError, match="require gradients"):
        fisher_blocks(ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_blocks():
    ctx = tiny_context(device="cuda")
    estimate = fisher_blocks(ctx, ("means", "log_scales"), n_samples=8)
    assert estimate.device.type == "cuda" and estimate.shape == (20, 6, 6)
    assert torch.isfinite(estimate).all()
    # Analytical check on the same device also verifies normalization and gradient routing.
    linear = replace(ctx, rasterize=linear_rasterize, cache={})
    direction = ctx.model.means.new_tensor([1.0, 2.0, -0.5])
    gradient = ctx.model.opacities.detach()[:, None] * direction
    exact = len(ctx.cams) * gradient[:, :, None] * gradient[:, None, :]
    torch.testing.assert_close(fisher_blocks(linear, n_samples=4), exact)


@pytest.mark.parametrize("name", ["fisher", "pup", "curvature", "s_geom", "s_geom_ranksum"])
def test_contract(name):
    """compute_signals(..., names=(name,)) -> shape (N,), float32, finite."""
    from frecon.signals import compute_signals
    ctx = tiny_context()
    for cam in ctx.cams:
        cam.image = (ctx.model.render(ctx.rasterize, cam, ctx.bg)["image"].detach() * 0.95).clamp(0, 1)
    score = compute_signals(ctx.model, ctx.cams, ctx.rasterize, ctx.bg, names=(name,))[name]
    assert score.shape == (20,) and score.dtype == torch.float32
    assert torch.isfinite(score).all() and not score.requires_grad


def test_curvature_orientation(monkeypatch):
    """A Gaussian seen by more views (larger F_i) gets a lower curvature score u_i."""
    from frecon.signals import s_geom
    ctx = tiny_context()
    monkeypatch.setattr(s_geom, "fisher_blocks", lambda c, *_: torch.eye(3).expand(c.model.n, 3, 3) * len(c.cams))
    torch.testing.assert_close(s_geom.curvature(ctx, eps=0.1), torch.full((20,), 3 / 2.1))
    assert (s_geom.curvature(ctx, eps=0.1) < s_geom.curvature(replace(ctx, cams=ctx.cams[:1]), eps=0.1)).all()


def test_diagonal_matches_full_blocks():
    from frecon.signals.second_order import fisher_diagonal
    ctx = tiny_context()
    params = tuple(ctx.model.params)
    diagonal = fisher_diagonal(ctx, params, 8)
    full = fisher_blocks(ctx, params, 8)
    assert diagonal.shape == (20, 14)
    torch.testing.assert_close(diagonal, full.diagonal(dim1=-2, dim2=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_large_gpu_eigenspectrum_has_bounded_workspace():
    from frecon.signals.second_order import damped_eigenvalues
    blocks = torch.eye(6, device="cuda").expand(4096, 6, 6).contiguous()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    eigenvalues = damped_eigenvalues(blocks, 0.1)
    torch.cuda.synchronize()
    torch.testing.assert_close(eigenvalues, torch.full((4096, 6), 1.1, device="cuda", dtype=torch.float64))
    assert torch.cuda.max_memory_allocated() - baseline < 1024 ** 3


def test_fisher_and_pup_formulas(monkeypatch):
    from frecon.signals import fisher, pup_sensitivity
    ctx = tiny_context()
    diagonal = torch.arange(1, 15).float().expand(20, 14)
    monkeypatch.setattr(fisher, "fisher_diagonal", lambda *_: diagonal)
    torch.testing.assert_close(fisher.compute(ctx, eps=0.1), -(diagonal.sum(-1) + 0.1).log())
    blocks = torch.diag(torch.arange(1, 7).float()).expand(20, 6, 6)
    monkeypatch.setattr(pup_sensitivity, "fisher_blocks", lambda *_: blocks)
    factors = torch.cat([torch.ones(20, 3), ctx.model.scales.detach().double().reciprocal()], -1)
    converted = blocks.double() * factors[:, :, None] * factors[:, None, :]
    expected = -torch.linalg.slogdet(converted + torch.eye(6) * 0.1).logabsdet.float()
    torch.testing.assert_close(pup_sensitivity.compute(ctx, eps=0.1), expected)
    log_score = pup_sensitivity.compute(ctx, eps=0.1, scale_basis="log_scale")
    assert not torch.equal(log_score, expected)
    # An entirely unobserved block remains finite after positive damping.
    monkeypatch.setattr(pup_sensitivity, "fisher_blocks", lambda *_: torch.zeros(20, 6, 6))
    torch.testing.assert_close(pup_sensitivity.compute(ctx, eps=0.1), torch.full((20,), -6 * torch.log(torch.tensor(0.1))))


def test_opposite_gradient_inconsistency_and_fallback(monkeypatch):
    from frecon.signals import s_geom
    ctx = tiny_context()
    def render(means, *args, **kwargs):
        return {"image": means.sum().reshape(1, 1, 1), "contrib": torch.ones(means.shape[0])}
    ctx.rasterize = render
    monkeypatch.setattr(s_geom, "photometric_loss", lambda pred, gt, _: pred.sum() * gt.flatten()[0])
    for cam, direction in zip(ctx.cams, [1.0, -1.0]):
        cam.image = torch.tensor([[[direction]]])
    score = s_geom.inconsistency(ctx, delta=1e-6)
    torch.testing.assert_close(score, torch.full((20,), 3e6))
    ctx.cams[1].image = torch.ones(1, 1, 1)
    same = s_geom.inconsistency(replace(ctx, cache={}), delta=1e-6)
    torch.testing.assert_close(same, torch.zeros(20))
    unknown = s_geom.inconsistency(replace(ctx, cams=ctx.cams[:1], cache={}))
    torch.testing.assert_close(unknown, torch.ones(20))


def test_rank_ties_and_invalid_regularization():
    from frecon.signals import fisher, pup_sensitivity, s_geom
    torch.testing.assert_close(s_geom.average_rank(torch.tensor([1., 1., 3.])), torch.tensor([0.25, 0.25, 1.]))
    torch.testing.assert_close(s_geom.average_rank(torch.ones(3)), torch.full((3,), 0.5))
    ctx = tiny_context()
    for function in (fisher.compute, pup_sensitivity.compute, s_geom.curvature):
        with pytest.raises(ValueError, match="eps"):
            function(ctx, eps=0)


def convergence_study(out_dir, device="cpu"):
    """Write actual estimator-vs-Jacobian results, not samples drawn directly from J."""
    import csv
    import json
    import platform
    import subprocess
    import sys
    import time
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for params in [("means",), ("means", "log_scales")]:
            ctx = tiny_context(device=device)
            exact = exact_blocks(ctx, params)
            for seed in range(4):
                for samples in [4, 64, 1024]:
                    if device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    estimate = fisher_blocks(replace(ctx, seed=seed, cache={}), params, samples)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    error = float((estimate - exact).norm() / exact.norm())
                    record = dict(params="+".join(params), probe_seed=seed, n_samples=samples,
                                  relative_frobenius_error=error, seconds=seconds)
                    if device == "cuda":
                        record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    records.append(record)
                    print(record, flush=True)
    finally:
        torch.set_num_threads(previous)

    with (out_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                    platform=platform.platform(), device=device, scene_seed=123,
                    n_gaussians=20, views=2, resolution=16, cpu_threads=1,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    working_tree=subprocess.check_output(["git", "status", "--short"], text=True),
                    command=" ".join(sys.argv))
    if device == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name()
    (out_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, params in zip(axes, ["means", "means+log_scales"]):
        grouped = [torch.tensor([r["relative_frobenius_error"] for r in records
                                 if r["params"] == params and r["n_samples"] == s]) for s in [4, 64, 1024]]
        means = torch.tensor([v.mean() for v in grouped])
        stds = torch.tensor([v.std() for v in grouped])
        ax.errorbar([4, 64, 1024], means, yerr=stds, fmt="o-", capsize=4, label="Mean +/- std (4 seeds)")
        ax.plot([4, 64, 1024], means[0] * torch.tensor([1.0, 0.25, 0.0625]), "--", label="1/sqrt(S) reference")
        ax.set(xscale="log", yscale="log", xlabel="Probes per training view", ylabel="Relative Frobenius error", title=params)
        ax.legend(fontsize=8)
        # Declared before the study: broad statistical gates, not per-draw monotonicity.
        assert means[-1] < 0.15 and means[-1] < means[0] * 0.25, (params, means)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results/second_order")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    convergence_study(args.out_dir, args.device)
