"""Per-Gaussian Gauss–Newton / Fisher blocks via a Hutchinson-style estimator.   Owner: Phát

    F_i = sum_views sum_pixels J_{p,i}^T J_{p,i},      J_{p,i} = dI_p / d theta_i

Estimator: for Rademacher v (one +-1 per pixel and channel),
    g = J^T v = d<v, I>/d theta      (one backward pass)
    E[g_i g_i^T] = F_i               because E[v v^T] = I
Average g_i g_i^T over n_samples draws per view and sum over views.

Blocks use the model's STORED parameters: "log_scales" means derivatives with respect to
log(scale), not activated scale. No damping, inverse, or score transformation is applied.
This is a rendered-image Gauss–Newton matrix, not the Hessian of the L1/SSIM training loss.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral

import torch

from .common import SignalContext


def fisher_blocks(ctx: SignalContext, params: tuple = ("means",), n_samples: int = 4) -> torch.Tensor:
    """Estimate sum_views J_i.T @ J_i without constructing a pixel Jacobian.

    ``params`` selects unique names from ``model.params`` in the requested order. Means
    give (N,3,3); means plus log_scales give (N,6,6), including their cross terms. Scalar
    parameter fields, e.g. opacity_logits (N,), are supported as one column per Gaussian.

    Only ctx.cams (training views) are used; their GT images are not needed. Invisible
    or unused parameters have zero information. Empty camera lists and frozen selected parameters
    are rejected. Accumulation is float32, or float64 for a float64 model.

    Probes have their own seeded generator per view, independent of global/shared RNG
    state and signal execution order. Sample budgets reuse the same probe prefixes.
    Results are detached and cached by ("fisher", params, n_samples, seed). As with
    view_statistics, use a NEW context after changing the model, cameras, or renderer;
    callers must not mutate cached blocks. Model parameters and .grad buffers are preserved.
    Memory holds only one view's render graph and the accumulated (N,D,D) blocks.
    """
    return _estimate(ctx, params, n_samples, diagonal=False)


def fisher_diagonal(ctx: SignalContext, params: tuple = ("means",), n_samples: int = 4) -> torch.Tensor:
    """(N,D) diagonal with the same probes as fisher_blocks, without allocating D*D blocks."""
    return _estimate(ctx, params, n_samples, diagonal=True)


def positive_scalar(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def finite_score(value):
    """Enforce the shared float32, finite score contract after stable calculations."""
    score = value.detach().float()
    if not torch.isfinite(score).all():
        raise ValueError("non-finite score; check input values and regularization parameters")
    return score


def damped_eigenvalues(blocks, eps):
    """Float64 PSD spectrum + damping; bound cuSOLVER workspace for many tiny GPU blocks.

    A single 28k-matrix eigvalsh launch can request ~15 GiB of workspace on CUDA. Batches of
    512 bound that scratch allocation, while preserving the independent block calculation.
    """
    positive_scalar(eps, "eps")
    blocks = blocks.double()
    symmetric = 0.5 * (blocks + blocks.transpose(-1, -2))
    if blocks.is_cuda and blocks.shape[0] > 512:
        eigenvalues = torch.cat([torch.linalg.eigvalsh(part) for part in symmetric.split(512)])
    else:
        eigenvalues = torch.linalg.eigvalsh(symmetric)
    return eigenvalues.clamp_min(0) + eps


@torch.enable_grad()
def _estimate(ctx, params, n_samples, diagonal):
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if isinstance(params, str):
        raise ValueError("params must be a non-empty sequence of parameter names")
    params = tuple(params)
    if not params or any(not isinstance(name, str) for name in params) or len(set(params)) != len(params):
        raise ValueError("params must contain unique parameter names")
    unknown = set(params) - ctx.model.params.keys()
    if unknown:
        raise ValueError(f"unknown model parameters: {sorted(unknown)}")
    if not ctx.cams:
        raise ValueError("fisher_blocks requires at least one training camera")
    selected = tuple(ctx.model.params[name] for name in params)
    if any(not p.requires_grad for p in selected):
        raise ValueError("selected model parameters must require gradients")

    n_samples = int(n_samples)
    key = ("fisher_diag" if diagonal else "fisher", params, n_samples, ctx.seed)
    if key in ctx.cache:
        return ctx.cache[key]

    n, device = ctx.model.n, ctx.model.means.device
    widths = tuple(math.prod(p.shape[1:]) for p in selected)
    dim = sum(widths)
    dtype = torch.float64 if ctx.model.means.dtype == torch.float64 else torch.float32
    blocks = torch.zeros((n, dim) if diagonal else (n, dim, dim), device=device, dtype=dtype)
    if n:
        for view_index, cam in enumerate(ctx.cams):
            # A stable domain-specific seed; neither Python's randomized hash nor ctx.generator.
            digest = hashlib.sha256(f"frecon:fisher:{ctx.seed}:{view_index}".encode()).digest()
            seed = int.from_bytes(digest[:8], "little") % (2 ** 63)
            generator = torch.Generator(device=device).manual_seed(seed)
            image = ctx.model.render(ctx.rasterize, cam, ctx.bg)["image"]
            if not image.requires_grad:
                continue  # e.g. a backend returning constant background for an empty view
            for sample in range(n_samples):
                probe = torch.empty_like(image).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                grads = torch.autograd.grad(
                    image, selected, grad_outputs=probe, allow_unused=True,
                    retain_graph=sample + 1 < n_samples, create_graph=False,
                )
                gradient = torch.cat([
                    torch.zeros(n, width, device=device, dtype=dtype) if grad is None
                    else grad.detach().reshape(n, width).to(dtype)
                    for grad, width in zip(grads, widths)
                ], dim=1)
                product = gradient.square() if diagonal else gradient[:, :, None] * gradient[:, None, :]
                blocks.add_(product, alpha=1.0 / n_samples)
            del image
    if not torch.isfinite(blocks).all():
        raise ValueError("Fisher block estimate contains NaN or Inf")
    ctx.cache[key] = blocks
    return blocks
