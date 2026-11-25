"""Offline floater ranking and reversible interventions on an RGB 3DGS snapshot."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from ..losses import psnr, ssim_map
from ..render.torch_rast import quat_to_rotmat, rasterize
from .d1_metrics import all_metrics


@dataclass
class Features:
    gaussian: torch.Tensor
    views: torch.Tensor
    observed: torch.Tensor
    fine: torch.Tensor
    coarse: torch.Tensor
    neighborhood: torch.Tensor

    @property
    def eligible(self):
        return self.observed.any(dim=1)


def xyz_array(points):
    return points.detach().cpu().numpy()


def neighborhood_graph(points, radius, k, device):
    """Degree-normalized adjacency and per-point geometry in a capped radius."""
    n = len(points)
    distance, index = cKDTree(points).query(points, k=range(1, min(k + 1, n) + 1),
                                          distance_upper_bound=radius)
    valid = (index != np.arange(n)[:, None]) & (index < n)
    row, slot = np.nonzero(valid)
    col = index[row, slot]
    degree = np.bincount(row, minlength=n)
    count = np.maximum(degree, 1)
    offset = points[col] - points[row]
    mean_offset = np.zeros_like(points)
    np.add.at(mean_offset, row, offset)
    mean_offset /= count[:, None] * radius
    mean_dist = np.bincount(row, weights=distance[row, slot], minlength=n) / (count * radius)
    stats = np.column_stack((degree / k, mean_dist, np.linalg.norm(mean_offset, axis=1)))
    graph = torch.sparse_coo_tensor(np.stack((row, col)), 1.0 / count[row], (n, n),
                                    device=device, dtype=torch.float32, check_invariants=True).coalesce()
    return graph, torch.as_tensor(stats, device=device, dtype=torch.float32)


@torch.no_grad()
def extract_features(model, train_cams, bg, extent):
    """Only snapshot parameters and the training camera list enter the ranker."""
    device = model.means.device
    log_scales, axes = model.scales.div(extent).clamp_min(1e-8).log().sort(dim=1)
    rotation = quat_to_rotmat(model.quats)
    gaussian = torch.cat((log_scales, model.opacities[:, None], log_scales[:, 1:] - log_scales[:, :1]), 1)
    tokens, support = [], []
    for cam in train_cams:
        out = model.render(rasterize, cam, bg, return_contrib=True)
        relative = cam.center - model.means
        direction = F.normalize(relative, dim=1)
        alignment = torch.bmm(direction[:, None], rotation).squeeze(1).abs().gather(1, axes)
        depth = ((model.means @ cam.R.T + cam.t)[:, 2] / extent).clamp(-8, 8)
        contrib = out["contrib"]
        radius = out["radii"]
        observed = contrib > 1e-3
        tokens.append(torch.column_stack((
            torch.log1p(1e4 * contrib / (cam.W * cam.H)), radius / min(cam.W, cam.H),
            contrib / (torch.pi * radius.square().clamp_min(1)), depth, alignment,
        )))
        support.append(observed)
    points = xyz_array(model.means / extent)
    fine, fine_stats = neighborhood_graph(points, radius=0.04, k=16, device=device)
    coarse, coarse_stats = neighborhood_graph(points, radius=0.12, k=32, device=device)
    return Features(gaussian, torch.stack(tokens, 1), torch.stack(support, 1), fine, coarse,
                    torch.cat((fine_stats, coarse_stats), dim=1))


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.GELU())


class FloaterClassifier(nn.Module):
    """One logit per Gaussian, with optional multi-view and spatial context."""

    def __init__(self, variant="hierarchical"):
        super().__init__()
        if variant not in ("point", "views", "hierarchical"):
            raise ValueError(f"Unknown classifier variant {variant!r}")
        self.variant = variant
        self.point = mlp(6, 32)
        if variant != "point":
            self.view = mlp(7, 16)
            self.fuse = mlp(32 + 16 * 2 + 1, 48)
        if variant == "hierarchical":
            self.fine = mlp(48 * 2 + 3, 48)
            self.coarse = mlp(48 * 2 + 3, 64)
        sizes = {"point": 32, "views": 48, "hierarchical": 48 + 48 + 64}
        self.logit = nn.Linear(sizes[variant], 1)

    def forward(self, data: Features):
        point = self.point(data.gaussian)
        if self.variant == "point":
            return self.logit(point).squeeze(1)
        view = self.view(data.views)
        observed = data.observed.unsqueeze(2)
        count = observed.sum(dim=1)
        mean = view.masked_fill(~observed, 0).sum(dim=1) / count.clamp_min(1)
        peak = view.masked_fill(~observed, -torch.inf).amax(dim=1).masked_fill(count == 0, 0)
        point = self.fuse(torch.cat((point, mean, peak, count / view.shape[1]), dim=1))
        if self.variant == "views":
            return self.logit(point).squeeze(1)
        fine = self.fine(torch.cat((point, torch.sparse.mm(data.fine, point), data.neighborhood[:, :3]), 1))
        coarse = self.coarse(torch.cat((fine, torch.sparse.mm(data.coarse, fine), data.neighborhood[:, 3:]), 1))
        return self.logit(torch.cat((point, fine, coarse), 1)).squeeze(1)


def distances(means, ref):
    """Unsigned distance in scene world units to the nearest reference sample."""
    d, _ = cKDTree(xyz_array(ref)).query(xyz_array(means))
    return torch.as_tensor(d, device=means.device, dtype=torch.float32)


def ranking_metrics(scores, d, tau):
    return all_metrics(xyz_array(scores), xyz_array(d > tau), xyz_array(d))


def select_candidates(scores, eligible, fraction, score_threshold=None):
    """At most round(fraction*N) candidates, breaking score ties by checkpoint row."""
    available = eligible.clone()
    if score_threshold is not None:
        available &= scores >= score_threshold
    ids = available.nonzero(as_tuple=True)[0]
    budget = max(1, round(fraction * scores.numel()))
    order = scores[ids].argsort(descending=True, stable=True)
    return ids[order[:budget]]


@torch.no_grad()
def reference_depth(ref, cam):
    coords = ref @ cam.R.T + cam.t
    z = coords[:, 2]
    uv = coords[:, :2] / z.clamp_min(1e-6)[:, None]
    x = (uv[:, 0] * cam.fx + cam.cx).floor().long()
    y = (uv[:, 1] * cam.fy + cam.cy).floor().long()
    valid = (z > 0.01) & (x >= 0) & (x < cam.W) & (y >= 0) & (y < cam.H)
    depth = z.new_full((cam.H * cam.W,), torch.inf)
    depth.scatter_reduce_(0, y[valid] * cam.W + x[valid], z[valid], reduce="amin")
    return depth.reshape(cam.H, cam.W)


@torch.no_grad()
def evaluate_mask(model, cams, bg, hidden, ref, d, ref_depths, tau, coverage_radius):
    """Rerender with zero opacity for selected rows. Includes per-view diagnostics."""
    opacity = model.opacities.clone()
    opacity[hidden] = 0
    active = opacity > 0.05
    if active.any():
        comp = distances(ref, model.means[active])
        coverage = float((comp <= coverage_radius).float().mean())
        completeness = float(comp.mean())
    else:
        coverage, completeness = 0.0, None
    far = d > tau
    images, per_view, depths = [], [], []
    float_weight = total_weight = n_pixels = 0
    for cam, gt_depth in zip(cams, ref_depths, strict=True):
        z = (model.means @ cam.R.T + cam.t)[:, 2]
        channels = torch.column_stack((model.colors, z))
        out = model.render(rasterize, cam, torch.cat((bg, bg.new_zeros(1))), features=channels,
                           opacities=opacity, return_contrib=True)
        image = out["image"][..., :3].clamp(0, 1)
        pred_depth = out["image"][..., 3] / out["alpha"].clamp_min(1e-8)
        valid = gt_depth.isfinite()
        supported = valid & (out["alpha"] >= 0.1) & ((pred_depth - gt_depth).abs() <= coverage_radius)
        fw = float(out["contrib"][far].sum())
        tw = float(out["contrib"].sum())
        float_weight += fw
        total_weight += tw
        n_pixels += cam.H * cam.W
        per_view.append(dict(view=cam.name, psnr=psnr(image, cam.image),
                             ssim=float(ssim_map(image, cam.image).mean()),
                             rgb_mae=float((image - cam.image).abs().mean()),
                             w_float=fw / tw if tw else None,
                             supported_pixels=int(supported.sum()), reference_pixels=int(valid.sum())))
        images.append(image.cpu())
        depths.append(supported.cpu())
    ref_pixels = sum(v["reference_pixels"] for v in per_view)
    metrics = dict(
        w_float=float_weight / total_weight if total_weight else None,
        float_weight_per_pixel=float_weight / n_pixels,
        alpha_per_pixel=total_weight / n_pixels,
        center_coverage=coverage,
        mean_completeness=completeness,
        depth_coverage=sum(v["supported_pixels"] for v in per_view) / ref_pixels if ref_pixels else None,
        psnr=float(np.mean([v["psnr"] for v in per_view])),
        ssim=float(np.mean([v["ssim"] for v in per_view])),
        rgb_mae=float(np.mean([v["rgb_mae"] for v in per_view])),
    )
    return metrics, per_view, torch.stack(images), torch.stack(depths)
