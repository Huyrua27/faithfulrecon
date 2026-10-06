"""D2 measurements: locality (D2.1), direction (D2.2), and the ingredients of the matched
contrast (D2.3), plus a D1 sanity check (floater labels, AUROC)."""
from __future__ import annotations

import torch

from .gaussians import GaussianModel, nearest_dist
from .losses import ssim_map, psnr


@torch.no_grad()
def render_views(model: GaussianModel, cams, rasterize, bg) -> torch.Tensor:
    return torch.stack([model.render(rasterize, c, bg)["image"].clamp(0, 1) for c in cams])


@torch.no_grad()
def region_masks(model: GaussianModel, selected: torch.Tensor, cams, rasterize, bg, thresh: float = 0.5):
    """Projected region of the selected Gaussians on each view: render them alone, at
    opacity 0.99, and threshold the accumulated alpha. Occlusion by other Gaussians is
    ignored on purpose: the region is where the selection *could* change the image."""
    sub = model.subset(selected)
    op = torch.full((sub.n,), 0.99, device=sub.means.device)
    feats = torch.zeros(sub.n, 1, device=sub.means.device)
    return torch.stack([sub.render(rasterize, c, bg[:1] * 0, features=feats, opacities=op)["alpha"] > thresh
                        for c in cams])


def image_delta(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor):
    """Mean absolute difference inside / outside `mask`. a, b: (V,H,W,3); mask (V,H,W)."""
    d = (a - b).abs().mean(-1)
    n_in, n_out = int(mask.sum()), int((~mask).sum())
    d_in = float(d[mask].sum() / max(n_in, 1))
    d_out = float(d[~mask].sum() / max(n_out, 1))
    return d_in, d_out, n_in, n_out


def locality_ratio(d_in: float, d_out: float) -> float:
    """Size-normalized locality ratio: inputs are already per-pixel means."""
    s = d_in + d_out
    return d_in / s if s > 0 else float("nan")


@torch.no_grad()
def heldout_errors(renders: torch.Tensor, cams) -> dict:
    l1, dssim, ps = [], [], []
    for img, cam in zip(renders, cams):
        l1.append(float((img - cam.image).abs().mean()))
        dssim.append(float(1 - ssim_map(img, cam.image).mean()))
        ps.append(psnr(img, cam.image))
    n = len(cams)
    return {"l1": sum(l1) / n, "dssim": sum(dssim) / n, "psnr": sum(ps) / n}


@torch.no_grad()
def neighborhood(points: torch.Tensor, centers: torch.Tensor, radius: float) -> torch.Tensor:
    """Mask of `points` within `radius` of any of `centers`."""
    return nearest_dist(points, centers) <= radius


@torch.no_grad()
def local_geometry(model_pts: torch.Tensor, ref_pts: torch.Tensor, centers: torch.Tensor, radius: float) -> dict:
    """Accuracy (model -> reference), completeness (reference -> model) and their mean
    (Chamfer), restricted to the neighborhood of `centers` (the removed Gaussians)."""
    m_loc = model_pts[neighborhood(model_pts, centers, radius)]
    r_loc = ref_pts[neighborhood(ref_pts, centers, radius)]
    acc = float(nearest_dist(m_loc, ref_pts).mean()) if m_loc.shape[0] else float("nan")
    comp = float(nearest_dist(r_loc, model_pts).mean()) if r_loc.shape[0] else float("nan")
    return {"acc": acc, "comp": comp, "cd": 0.5 * (acc + comp), "n_model": int(m_loc.shape[0]),
            "n_ref": int(r_loc.shape[0])}


@torch.no_grad()
def displacement_split(means_a: torch.Tensor, means_b: torch.Tensor, near: torch.Tensor):
    """Mean center displacement between two models with the same Gaussians, split into
    Gaussians near the intervention (`near`) and the rest."""
    d = (means_a - means_b).norm(dim=-1)
    d_in = float(d[near].mean()) if near.any() else float("nan")
    d_out = float(d[~near].mean()) if (~near).any() else float("nan")
    return d_in, d_out


@torch.no_grad()
def floater_labels(model: GaussianModel, ref_pts: torch.Tensor, heldout_contrib: torch.Tensor,
                   tau: float, min_opacity: float = 0.05, min_contrib: float = 1e-2) -> torch.Tensor:
    """Outcome label f_i: far from the reference surface AND visible on held-out views."""
    far = nearest_dist(model.means.detach(), ref_pts) > tau
    return far & (model.opacities.detach() > min_opacity) & (heldout_contrib > min_contrib)


def auroc(score: torch.Tensor, label: torch.Tensor) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    s = score.double()
    order = torch.argsort(s)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, device=s.device, dtype=s.dtype)
    # average ranks over ties
    uniq, inv, counts = torch.unique(s, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = torch.zeros_like(uniq).index_add(0, inv, ranks)
        ranks = (sums / counts)[inv]
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))
