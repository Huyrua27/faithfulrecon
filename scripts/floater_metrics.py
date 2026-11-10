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


def load_eval_scene(cfg, dev):
    sc = cfg["scene"]
    assert sc.get("type") == "dtu", "floater_metrics.py supports DTU scenes only"
    cams, masks, scale_mat = load_dtu_views(sc["root"], sc.get("downscale", 4))
    tr, te = split_views(cams, sc["n_train_views"], sc["n_test_views"])
    ref = reference_points(sc.get("stl_ply"), scale_mat, cams, masks, sc.get("min_mask_views", 1),
                           sc.get("max_ref_points", 200_000), sc.get("ref_cache")).to(dev)
    views = []
    for i in tr:
        cam = cams[i].to(dev)
        views.append((cam, torch.from_numpy(masks[i]).to(dev), zbuffer(ref, cam)))
    return views, [cams[i].to(dev) for i in te], ref


@torch.no_grad()
def metrics(model, views, test_cams, ref, rasterize, bg, tol):
    means = model.means.detach()
    d = nearest_dist(means, ref)
    off01 = d > 0.1
    med = depth_side(means, views, tol)
    contrib = torch.zeros(model.n, device=means.device)
    for cam in test_cams:
        contrib += model.render(rasterize, cam, bg, return_contrib=True)["contrib"]
    solid = model.opacities.detach().reshape(-1) > 0.05
    err = heldout_errors(render_views(model, test_cams, rasterize, bg), test_cams)
    return {
        "n_gaussians": model.n,
        "off_003": float((d > 0.03).float().mean()),
        "off_01": float(off01.float().mean()),
        "front_01": float(((med < -tol) & off01).sum() / off01.sum().clamp_min(1)),
        "wfloat_01": float(contrib[off01].sum() / contrib.sum().clamp_min(1e-12)),
        "acc": float(d[solid].mean()),
        "comp": float(nearest_dist(ref, means[solid]).mean()),
        "psnr": err["psnr"], "dssim": err["dssim"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--out", default="results/photometric/floater_metrics.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows, cache = [], {}
    for rd in map(Path, args.run_dirs):
        cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
        rasterize = get_rasterizer(cfg.get("renderer", "torch"))
        bg = torch.ones(3, device=dev)
        key = cfg["scene"]["root"]
        if key not in cache:   # evaluation scene ignores `photometric`: original appearance
            cache = {key: load_eval_scene(cfg, dev)}
        views, test_cams, ref = cache[key]
        photo = cfg["scene"].get("photometric") or {}
        for seed in args.seeds:
            ckpt = rd / f"seed_{seed}" / "base.pt"
            if not ckpt.exists():
                print(f"skip {ckpt} (missing)")
                continue
            model = Trainer.load(ckpt, TrainConfig(**cfg["train"]), rasterize, bg, dev).model
            row = {"run": rd.name, "scan": Path(key).name, "seed": seed,
                   "exposure_stops": photo.get("exposure_stops", 0.0), "wb": photo.get("wb", 0.0),
                   "gamma": photo.get("gamma", 0.0)}
            row.update(metrics(model, views, test_cams, ref, rasterize, bg, args.tol))
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
