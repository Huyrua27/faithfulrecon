"""Gaussian scene representation (RGB colors, i.e. SH degree 0)."""
from __future__ import annotations

import torch

PARAM_NAMES = ("means", "log_scales", "quats", "opacity_logits", "color_logits")


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1 - x))


class GaussianModel:
    def __init__(self, means, log_scales, quats, opacity_logits, color_logits):
        self.params = {
            "means": torch.nn.Parameter(means.contiguous()),
            "log_scales": torch.nn.Parameter(log_scales.contiguous()),
            "quats": torch.nn.Parameter(quats.contiguous()),
            "opacity_logits": torch.nn.Parameter(opacity_logits.contiguous()),
            "color_logits": torch.nn.Parameter(color_logits.contiguous()),
        }

    # ---- construction ------------------------------------------------------------
    @classmethod
    def from_points(cls, points: torch.Tensor, colors: torch.Tensor, init_opacity: float = 0.1):
        """3DGS-style init: isotropic scale = mean distance to 3 nearest neighbors."""
        n = points.shape[0]
        d = _knn_mean_dist(points, k=3).clamp_min(1e-7)
        quats = torch.zeros(n, 4, device=points.device)
        quats[:, 0] = 1.0
        return cls(points, torch.log(d)[:, None].repeat(1, 3), quats,
                   inverse_sigmoid(torch.full((n,), init_opacity, device=points.device)),
                   inverse_sigmoid(colors))

    @classmethod
    def random_in_box(cls, n: int, half_extent: float, generator: torch.Generator, device):
        pts = (torch.rand(n, 3, generator=generator, device=device) * 2 - 1) * half_extent
        cols = torch.rand(n, 3, generator=generator, device=device)
        return cls.from_points(pts, cols)

    # ---- activated views -------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.params["means"].shape[0]

    @property
    def means(self):
        return self.params["means"]

    @property
    def scales(self):
        return torch.exp(self.params["log_scales"])

    @property
    def quats(self):
        return self.params["quats"]

    @property
    def opacities(self):
        return torch.sigmoid(self.params["opacity_logits"])

    @property
    def colors(self):
        return torch.sigmoid(self.params["color_logits"])

    def render(self, rasterize, cam, bg, features=None, opacities=None, **kw):
        return rasterize(self.means, self.quats, self.scales,
                         self.opacities if opacities is None else opacities,
                         self.colors if features is None else features, cam, bg, **kw)

    # ---- io ------------------------------------------------------------------------
    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.params.items()}

    @classmethod
    def from_state_dict(cls, sd, device=None):
        sd = {k: (v.to(device) if device is not None else v) for k, v in sd.items()}
        return cls(*(sd[k] for k in PARAM_NAMES))

    def subset(self, keep: torch.Tensor) -> "GaussianModel":
        return GaussianModel(*(self.params[k].detach()[keep].clone() for k in PARAM_NAMES))


@torch.no_grad()
def _knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 1024) -> torch.Tensor:
    out = []
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i + chunk], points)
        d = torch.topk(d, k + 1, largest=False).values[:, 1:]
        out.append(d.mean(-1))
    return torch.cat(out)


@torch.no_grad()
def nearest_dist(query: torch.Tensor, ref: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Distance from each query point to its nearest reference point."""
    if ref.shape[0] == 0:
        return torch.full((query.shape[0],), float("inf"), device=query.device)
    out = []
    for i in range(0, query.shape[0], chunk):
        out.append(torch.cdist(query[i:i + chunk], ref).min(dim=1).values)
    return torch.cat(out) if out else torch.zeros(0, device=query.device)
