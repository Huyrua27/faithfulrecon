"""Photometric losses and image metrics. Images are (H, W, 3) in [0, 1]."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_WINDOWS: dict = {}


def _window(size: int, sigma: float, device, dtype):
    key = (size, sigma, device, dtype)
    if key not in _WINDOWS:
        g = torch.arange(size, device=device, dtype=dtype) - size // 2
        g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _WINDOWS[key] = (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()
    return _WINDOWS[key]


def ssim_map(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Per-pixel SSIM (H, W), averaged over channels."""
    x = x.permute(2, 0, 1)[None]
    y = y.permute(2, 0, 1)[None]
    w = _window(size, sigma, x.device, x.dtype)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s[0].mean(0)


def photometric_loss(pred, gt, lambda_dssim: float = 0.2):
    l1 = (pred - gt).abs().mean()
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim_map(pred, gt).mean())


def psnr(pred, gt) -> float:
    mse = ((pred - gt) ** 2).mean().clamp_min(1e-10)
    return float(-10 * torch.log10(mse))


class LPIPS:
    """Lazy LPIPS(alex). Returns None if the package or its weights are unavailable."""

    def __init__(self, device):
        self.device = device
        self._net = None
        self._failed = False

    def __call__(self, pred, gt):
        if self._failed:
            return None
        if self._net is None:
            try:
                import lpips
                self._net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
            except Exception:
                self._failed = True
                return None
        with torch.no_grad():
            a = pred.permute(2, 0, 1)[None] * 2 - 1
            b = gt.permute(2, 0, 1)[None] * 2 - 1
            return float(self._net(a, b))
