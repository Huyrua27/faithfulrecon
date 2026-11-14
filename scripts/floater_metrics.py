"""Floater and geometry metrics of trained DTU base models (one row per run_dir x seed).

python scripts/floater_metrics.py --run_dirs runs/dtu_scan24_v8 runs/photo_strong_dtu_scan24_v8 \
    --seeds 0 1 2 --out results/photometric/floater_metrics.csv

Metrics (normalized units; d = distance from a Gaussian mean to the nearest reference point):
  off_003, off_01   fraction of Gaussians with d > 0.03 / d > 0.1
  front_01          fraction of Gaussians with d > 0.1 that lie IN FRONT of the reference surface
                    (median over training views of z_gauss - z_surface < -tol)
  wfloat_01         share of held-out blending weight carried by Gaussians with d > 0.1
                    (the floaters that actually render)
  acc, comp         mean distance Gaussian -> reference and reference -> Gaussian
                    (Gaussians with opacity > 0.05)
  psnr, dssim       held-out views, ORIGINAL (unperturbed) appearance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.camera import farthest_point_indices  # noqa: E402
from frecon.data.dtu import load_dtu_views, reference_points, split_views  # noqa: E402
from frecon.gaussians import nearest_dist  # noqa: E402
from frecon.measure import heldout_errors, render_views  # noqa: E402
from frecon.render import get_rasterizer  # noqa: E402
from frecon.train import TrainConfig, Trainer  # noqa: E402


def project(P, cam):
    Xc = P @ cam.R.T + cam.t
    z = Xc[:, 2]
    u = torch.floor(cam.fx * Xc[:, 0] / z.clamp_min(1e-6) + cam.cx).long()
    v = torch.floor(cam.fy * Xc[:, 1] / z.clamp_min(1e-6) + cam.cy).long()
    ok = (z > 1e-4) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
    return z, u.clamp(0, cam.W - 1), v.clamp(0, cam.H - 1), ok


def zbuffer(ref, cam):
    z, u, v, ok = project(ref, cam)
    zb = torch.full((cam.H * cam.W,), float("inf"), device=ref.device)
    zb = zb.scatter_reduce(0, v[ok] * cam.W + u[ok], z[ok], "amin")
    return -F.max_pool2d(-zb.view(1, 1, cam.H, cam.W), 3, 1, 1)[0, 0]   # close small holes


@torch.no_grad()
def depth_side(means, views, tol):
    diffs = []
    for cam, mask, zb in views:
        z, u, v, ok = project(means, cam)
        zs = zb[v, u]
        ok &= mask[v, u] & torch.isfinite(zs)
        d = torch.full_like(z, float("nan"))
        d[ok] = z[ok] - zs[ok]
        diffs.append(d)
    return torch.nanmedian(torch.stack(diffs, 1), 1).values   # nan where no valid view


def split_eval(cams, n_train: int, n_test: int, test_pool: int = 0):
    """Training views as in training (FPS, nested in N). With test_pool > 0 the held-out views are
    taken outside the first max(n_train, test_pool) FPS cameras, so runs with any N <= test_pool share
    the same held-out views."""
    if test_pool <= 0:
        return split_views(cams, n_train, n_test)
    centers = torch.stack([c.center for c in cams])
    tr = farthest_point_indices(centers, n_train)
    pool = farthest_point_indices(centers, max(n_train, test_pool))
    rest = [i for i in range(len(cams)) if i not in pool]
    step = max(len(rest) // max(n_test, 1), 1)
    return tr, rest[::step][:n_test]


def strict_reference(sc, cams, masks, scale_mat, frac: float, seed: int = 0) -> torch.Tensor:
    """Reference points that lie inside the object mask in at least `frac` of the views whose image
    they project into (the default crop keeps points inside the mask of >= 1 view, which retains
    table / background points). Cached next to the default cache."""
    cache = Path(sc["ref_cache"]).with_name(f"ref_points_strict{int(round(frac * 100))}.npy")
    if cache.exists():
        return torch.from_numpy(np.load(cache)).float()
    from frecon.data.blender import _read_ply_xyz
    pts_w = _read_ply_xyz(Path(sc["stl_ply"])).numpy().astype(np.float64)
    inv = np.linalg.inv(scale_mat)
    pts = (pts_w @ inv[:3, :3].T) + inv[:3, 3]
    pts = pts[np.linalg.norm(pts, axis=1) < 1.5]
    n_in, n_frame = np.zeros(len(pts), np.int32), np.zeros(len(pts), np.int32)
    for cam, m in zip(cams, masks):
        R, t = cam.R.numpy().astype(np.float64), cam.t.numpy().astype(np.float64)
        pc = pts @ R.T + t
        z = pc[:, 2]
        ok = z > 1e-6
        u = np.floor(cam.fx * pc[:, 0] / np.where(ok, z, 1) + cam.cx).astype(np.int64)
        v = np.floor(cam.fy * pc[:, 1] / np.where(ok, z, 1) + cam.cy).astype(np.int64)
        inb = ok & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        hit = np.zeros(len(pts), dtype=bool)
        hit[inb] = m[v[inb], u[inb]]
        n_frame += inb
        n_in += hit
    keep = (n_frame > 0) & (n_in >= frac * n_frame)
    print(f"strict reference ({frac:.2f}): kept {keep.sum()}/{len(pts)} points", flush=True)
    pts = pts[keep]
    mx = sc.get("max_ref_points", 200_000)
    if len(pts) > mx:
        pts = pts[np.random.default_rng(seed).choice(len(pts), mx, replace=False)]
    pts = pts.astype(np.float32)
    np.save(cache, pts)
    return torch.from_numpy(pts)


def load_eval_scene(cfg, dev, ref_mode: str = "default", strict_frac: float = 0.8, test_pool: int = 0):
    sc = cfg["scene"]
    assert sc.get("type") == "dtu", "floater_metrics.py supports DTU scenes only"
    cams, masks, scale_mat = load_dtu_views(sc["root"], sc.get("downscale", 4))
    tr, te = split_eval(cams, sc["n_train_views"], sc["n_test_views"], test_pool)
    if ref_mode == "strict":
        ref = strict_reference(sc, cams, masks, scale_mat, strict_frac).to(dev)
    else:
        ref = reference_points(sc.get("stl_ply"), scale_mat, cams, masks, sc.get("min_mask_views", 1),
                               sc.get("max_ref_points", 200_000), sc.get("ref_cache")).to(dev)
    views = []
    for i in tr:
        cam = cams[i].to(dev)
        views.append((cam, torch.from_numpy(masks[i]).to(dev), zbuffer(ref, cam)))
    return views, [cams[i].to(dev) for i in te], ref


def tau_tag(tau: float) -> str:
    """0.03 -> '003', 0.1 -> '01', 0.15 -> '015' (keeps the historical column names)."""
    return f"{tau:g}".replace("0.", "0", 1)


@torch.no_grad()
def metrics(model, views, test_cams, ref, rasterize, bg, tol, taus=(0.03, 0.05, 0.1, 0.15)):
    """W_float(tau) = sum_{k,p} sum_{i: d_i > tau} w_ik(p) / sum_{k,p} sum_i w_ik(p) over all pixels of
    the held-out views (torch renderer: exact blending weights, contributions < 1/255 dropped)."""
    means = model.means.detach()
    opac = model.opacities.detach().reshape(-1)
    d = nearest_dist(means, ref)
    med = depth_side(means, views, tol)
    contrib = torch.zeros(model.n, device=means.device)
    for cam in test_cams:
        contrib += model.render(rasterize, cam, bg, return_contrib=True)["contrib"]
    solid = opac > 0.05
    err = heldout_errors(render_views(model, test_cams, rasterize, bg), test_cams)
    out = {"n_gaussians": model.n}
    for tau in taus:
        far = d > tau
        t = tau_tag(tau)
        out[f"off_{t}"] = float(far.float().mean())
        out[f"wfloat_{t}"] = float(contrib[far].sum() / contrib.sum().clamp_min(1e-12))
        out[f"n_off_{t}"] = int(far.sum())                      # count, not a fraction
        out[f"opsum_off_{t}"] = float(opac[far].sum())          # total opacity of far Gaussians
        out[f"contrib_off_{t}"] = float(contrib[far].sum())     # absolute held-out weight
    off01 = d > 0.1
    out.update({
        "front_01": float(((med < -tol) & off01).sum() / off01.sum().clamp_min(1)),
        "contrib_total": float(contrib.sum()),
        "acc": float(d[solid].mean()),
        "comp": float(nearest_dist(ref, means[solid]).mean()),
        "psnr": err["psnr"], "dssim": err["dssim"],
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.03, 0.05, 0.1, 0.15],
                    help="distance thresholds (normalized units, scene ~ unit sphere) for the sweep")
    ap.add_argument("--out", default="results/photometric/floater_metrics.csv")
    ap.add_argument("--ref_mode", choices=["default", "strict"], default="default",
                    help="strict: keep reference points inside the mask in >= --strict_frac of the views "
                         "they project into")
    ap.add_argument("--strict_frac", type=float, default=0.8)
    ap.add_argument("--test_pool", type=int, default=0,
                    help="> 0: held-out views outside the first max(N, test_pool) FPS cameras (same views "
                         "for every N <= test_pool)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, cache = [], {}
    for rd in map(Path, args.run_dirs):
        cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
        rasterize = get_rasterizer(cfg.get("renderer", "torch"))
        bg = torch.ones(3, device=dev)
        key = (cfg["scene"]["root"], cfg["scene"]["n_train_views"])
        if key not in cache:   # evaluation scene ignores `photometric`: original appearance
            cache = {key: load_eval_scene(cfg, dev, args.ref_mode, args.strict_frac, args.test_pool)}
        views, test_cams, ref = cache[key]
        photo = cfg["scene"].get("photometric") or {}
        for seed in args.seeds:
            ckpt = rd / f"seed_{seed}" / "base.pt"
            if not ckpt.exists():
                print(f"skip {ckpt} (missing)")
                continue
            model = Trainer.load(ckpt, TrainConfig(**cfg["train"]), rasterize, bg, dev).model
            tc, ic = cfg.get("train", {}), cfg.get("init", {})
            row = {"run": rd.name, "scan": Path(key[0]).name, "seed": seed, "N": key[1],
                   "exposure_stops": photo.get("exposure_stops", 0.0), "wb": photo.get("wb", 0.0),
                   "gamma": photo.get("gamma", 0.0), "photo_seed": photo.get("seed", 0),
                   "photo_mode": photo.get("mode", "isp"), "appearance": tc.get("appearance", "none"),
                   "lr_appearance": tc.get("lr_appearance", 1e-3), "sh_degree": tc.get("sh_degree", 0),
                   "cap": bool(tc.get("cap_scale_after_densify", False)), "init": ic.get("type", "random"),
                   "ref_mode": args.ref_mode, "test_pool": args.test_pool}
            row.update(metrics(model, views, test_cams, ref, rasterize, bg, args.tol, args.taus))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
