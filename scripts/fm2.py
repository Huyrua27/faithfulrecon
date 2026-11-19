"""Evaluation v2 (strict reference, 16 shared held-out views), one row per run_dir x seed (x snapshot).

Always renders with the torch rasterizer (exact blending weights), whatever renderer was used for training.
Adds to floater_metrics.metrics:
  wsig_01       W_float with a shape-aware weight: per Gaussian, fraction of 7 sigma points (centre and
                centre +- 1 sigma along the 3 principal axes) farther than 0.1 from the reference
  psnr_aligned  held-out PSNR after a per-view, per-channel affine colour fit (removes a global colour shift)
With --snaps: light metrics (W_float(0.1), off(0.1), #Gaussians, PSNR) for every snap_XXXXX.pt and base.pt.

python scripts/fm2.py --run_dirs runs/v2_* --seeds 0 1 2 --out results/v2/x.csv [--snaps]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from floater_metrics import load_eval_scene, metrics  # noqa: E402
from frecon.gaussians import GaussianModel, nearest_dist  # noqa: E402
from frecon.measure import heldout_errors, render_views  # noqa: E402
from frecon.render import get_rasterizer  # noqa: E402
from frecon.render.torch_rast import quat_to_rotmat  # noqa: E402
from frecon.train import TrainConfig, Trainer  # noqa: E402

NAME = re.compile(r"(photo|v2|v3|v5)_(none|mild|strongng|strongs1|strongs2|strong|pac|pa|pb|pc)(.*)_dtu_scan(\d+)_v(\d+)$")


def parse(name: str) -> dict:
    m = NAME.match(name)
    if not m:
        return {"prefix": "?", "level": "?", "tags": name}
    return {"prefix": m[1], "level": m[2], "tags": m[3]}


@torch.no_grad()
def contrib(model, cams, ras, bg):
    c = torch.zeros(model.n, device=model.means.device)
    for cam in cams:
        c += model.render(ras, cam, bg, return_contrib=True)["contrib"]
    return c


@torch.no_grad()
def sigma_far(model, ref, tau=0.1):
    mu, s = model.means.detach(), model.scales.detach()
    R = quat_to_rotmat(model.quats.detach())
    far = (nearest_dist(mu, ref) > tau).float()
    for m in range(3):
        ax = R[:, :, m] * s[:, m:m + 1]
        for sg in (-1.0, 1.0):
            far += (nearest_dist(mu + sg * ax, ref) > tau).float()
    return far / 7


@torch.no_grad()
def aligned_psnr(model, cams, ras, bg):
    out = []
    for cam in cams:
        x = model.render(ras, cam, bg)["image"].clamp(0, 1).reshape(-1, 3)
        y = cam.image.reshape(-1, 3)
        fit = torch.empty_like(x)
        for ch in range(3):
            A = torch.stack([x[:, ch], torch.ones_like(x[:, ch])], 1)
            sol = torch.linalg.lstsq(A.cpu().double(), y[:, ch:ch + 1].cpu().double()).solution
            fit[:, ch] = (A @ sol.to(A).float())[:, 0]
        mse = ((fit.clamp(0, 1) - y) ** 2).mean()
        out.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
    return float(np.mean(out))


def light(model, test_cams, ref, ras, bg):
    cb = contrib(model, test_cams, ras, bg)
    d = nearest_dist(model.means.detach(), ref)
    far = d > 0.1
    err = heldout_errors(render_views(model, test_cams, ras, bg), test_cams)
    return {"wfloat_01": float(cb[far].sum() / cb.sum().clamp_min(1e-12)), "off_01": float(far.float().mean()),
            "n_gaussians": model.n, "psnr": err["psnr"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--snaps", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ras, bg = get_rasterizer("torch"), torch.ones(3, device=dev)
    rows, cache = [], {}
    for rd in map(Path, args.run_dirs):
        cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
        sc, tc, ic = cfg["scene"], cfg.get("train", {}), cfg.get("init", {})
        key = (sc["root"], sc["n_train_views"])
        if key not in cache:
            cache = {key: load_eval_scene(cfg, dev, "strict", 0.8, 16)}
        views, test_cams, ref = cache[key]
        photo = sc.get("photometric") or {}
        info = dict(run=rd.name, scan=Path(sc["root"]).name, N=sc["n_train_views"], **parse(rd.name),
                    exposure_stops=photo.get("exposure_stops", 0.0), photo_seed=photo.get("seed", 0),
                    perm=str(photo.get("perm", "")), clip=bool(photo.get("clip", True)),
                    photo_mode=photo.get("mode", "isp"), appearance=tc.get("appearance", "none"),
                    lr_appearance=tc.get("lr_appearance", 1e-3), sh_degree=tc.get("sh_degree", 0),
                    sat_mask=bool(tc.get("sat_mask", False)), warmup=int(tc.get("warmup_geom", 0)),
                    init=ic.get("type", "random"), renderer=cfg.get("renderer", "torch"),
                    iterations=tc.get("iterations", 7000))
        for seed in args.seeds:
            ck = rd / f"seed_{seed}" / "base.pt"
            if not ck.exists():
                continue
            if args.snaps:
                for sp in sorted((rd / f"seed_{seed}").glob("snap_*.pt")) + [ck]:
                    if sp == ck:
                        model, step = Trainer.load(ck, TrainConfig(**tc), ras, bg, dev).model, info["iterations"]
                    else:
                        d = torch.load(sp, map_location=dev, weights_only=False)
                        model, step = GaussianModel.from_state_dict(d["params"], dev), int(d["step"])
                    rows.append(dict(info, seed=seed, step=step, **light(model, test_cams, ref, ras, bg)))
                    del model
                print(rd.name, seed, "snapshots done", flush=True)
            else:
                model = Trainer.load(ck, TrainConfig(**tc), ras, bg, dev).model
                row = dict(info, seed=seed, step=info["iterations"])
                row.update(metrics(model, views, test_cams, ref, ras, bg, 0.03, (0.05, 0.1)))
                cb = contrib(model, test_cams, ras, bg)
                row["wsig_01"] = float((cb * sigma_far(model, ref)).sum() / cb.sum().clamp_min(1e-12))
                row["psnr_aligned"] = aligned_psnr(model, test_cams, ras, bg)
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                       if k in ("run", "seed", "wfloat_01", "wsig_01", "psnr", "psnr_aligned")}, flush=True)
                del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
