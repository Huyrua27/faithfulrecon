"""Parity between the pure-PyTorch rasterizer and the optional gsplat backend (images + gradients).
Skipped when gsplat or CUDA is unavailable. Run on Colab before using `renderer: gsplat`.

Tolerances are loose on purpose: gsplat sorts per tile and terminates rays early (T < 1e-4), the
torch backend sorts per pixel and keeps all contributions. Mean differences must be small; a large
difference means a convention mismatch (pixel centers, camera axes) and gsplat must not be used.
"""
import pytest
import torch

from frecon.camera import orbit_cameras
from frecon.render import torch_rast

gsplat = pytest.importorskip("gsplat")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gsplat_matches_torch():
    from frecon.render.gsplat_rast import rasterize as g_rast

    torch.manual_seed(0)
    dev = "cuda"
    n = 2000
    means = ((torch.rand(n, 3) * 2 - 1) * 0.6).to(dev)
    quats = torch.randn(n, 4, device=dev)
    scales = (torch.rand(n, 3) * 0.03 + 0.005).to(dev)
    opac = (torch.rand(n) * 0.9 + 0.05).to(dev)
    feat = torch.rand(n, 3, device=dev)
    cam = orbit_cameras(1, 3.0, 128, 128, 40.0)[0].to(dev)
    bg = torch.ones(3, device=dev)

    outs, grads = [], []
    for r in (torch_rast.rasterize, g_rast):
        ps = [x.clone().requires_grad_(True) for x in (means, quats, scales, opac, feat)]
        img = r(*ps, cam, bg)["image"]
        img.sum().backward()
        outs.append(img.detach())
        grads.append([p.grad.clone() for p in ps])
    diff = (outs[0] - outs[1]).abs()
    print(f"image |diff| mean {diff.mean():.2e} max {diff.max():.2e}")
    assert diff.mean() < 5e-3
    for name, a, b in zip(["means", "quats", "scales", "opac", "feat"], *grads):
        rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        print(f"grad {name}: rel diff {rel:.2e}")
        assert rel < 0.1, name
