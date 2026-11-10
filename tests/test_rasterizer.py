"""Compare the pair-based rasterizer with a dense per-Gaussian reference (image and gradients).

Run: python -m tests.test_rasterizer
"""
import torch

from frecon.camera import look_at, focal_from_fov, Camera
from frecon.render import torch_rast


def dense_reference(means, quats, scales, opac, feat, cam, bg):
    m2d, con, radii, depth, valid = torch_rast.project(means, quats, scales, cam)
    order = torch.argsort(depth)
    ys, xs = torch.meshgrid(torch.arange(cam.H, device=means.device), torch.arange(cam.W, device=means.device), indexing="ij")
    pix = torch.stack([xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5], -1).float()  # (P,2)
    out = torch.zeros(pix.shape[0], feat.shape[1], device=means.device)
    T = torch.ones(pix.shape[0], device=means.device)
    for i in order.tolist():
        if not valid[i]:
            continue
        d = pix - m2d[i]
        # only pixels inside the square 3-sigma footprint, as in the pair enumeration
        inside = (d[:, 0].abs() <= radii[i]) & (d[:, 1].abs() <= radii[i])
        pw = -0.5 * (con[i, 0] * d[:, 0] ** 2 + con[i, 2] * d[:, 1] ** 2) - con[i, 1] * d[:, 0] * d[:, 1]
        a = (opac[i] * torch.exp(pw.clamp_max(0))).clamp_max(0.99)
        a = torch.where((pw <= 0) & (a >= 1 / 255) & inside, a, torch.zeros_like(a))
        out = out + (a * T)[:, None] * feat[i]
        T = T * (1 - a)
    return (out + T[:, None] * bg).reshape(cam.H, cam.W, -1)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, S = 150, 40
    means = (torch.rand(n, 3, device=dev) * 2 - 1) * 0.6
    quats = torch.randn(n, 4, device=dev)
    scales = torch.rand(n, 3, device=dev) * 0.08 + 0.01
    opac = torch.rand(n, device=dev) * 0.9 + 0.05
    feat = torch.rand(n, 3, device=dev)
    R, t = look_at(torch.tensor([2.5, 0.4, 0.8]), torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]))
    f = focal_from_fov(45, S)
    cam = Camera(R=R.to(dev), t=t.to(dev), fx=f, fy=f, cx=S / 2, cy=S / 2, W=S, H=S)
    bg = torch.ones(3, device=dev)

    ps = [x.clone().requires_grad_(True) for x in (means, quats, scales, opac, feat)]
    img = torch_rast.rasterize(*ps, cam, bg)["image"]
    w = torch.rand_like(img)
    (img * w).sum().backward()
    g_fast = [p.grad.clone() for p in ps]

    ps2 = [x.clone().requires_grad_(True) for x in (means, quats, scales, opac, feat)]
    ref = dense_reference(*ps2, cam, bg)
    (ref * w).sum().backward()
    g_ref = [p.grad for p in ps2]

    err = (img - ref).abs().max().item()
    print(f"max |image diff| = {err:.2e}")
    ok = err < 1e-4
    for name, a, b in zip(["means", "quats", "scales", "opac", "feat"], g_fast, g_ref):
        rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
        print(f"grad {name:7s} rel err = {rel:.2e}")
        ok &= rel < 1e-3
    print("PASS" if ok else "FAIL")
    return ok


def test_matches_dense_reference():
    assert main()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
