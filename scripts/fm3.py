"""Evaluation v3 = fm2 (W_float, W_float^Sigma, direct / aligned PSNR) + geometry metrics independent of W_float.

Geometry: expected depth is rendered (torch rasterizer) for the N training and 16 held-out views, pixels inside
the object mask with alpha > 0.5 are back-projected and merged (at most 300k points), and compared with the strict
STL reference:
  acc   mean distance prediction -> reference      comp  mean distance reference -> prediction
  chamfer = (acc + comp) / 2                       fscore_XX  F-score at threshold 0.0XX (normalized units)

python scripts/fm3.py --run_dirs runs/v3_* --seeds 0 1 2 --out results/v3/x.csv
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

import fm2  # noqa: E402
from floater_metrics import load_eval_scene, metrics, split_eval  # noqa: E402
from frecon.data.dtu import load_dtu_views  # noqa: E402
from frecon.gaussians import nearest_dist  # noqa: E402
from frecon.render import get_rasterizer  # noqa: E402
from frecon.train import TrainConfig, Trainer  # noqa: E402

NAME = re.compile(r"(photo|v2|v3|v5)_(none|mild|strongng|strongs1|strongs2|strong|pac|pa|pb|pc)(.*)_dtu_scan(\d+)_v(\d+)$")


def parse(name: str) -> dict:
    m = NAME.match(name)
    return {"prefix": m[1], "level": m[2], "tags": m[3]} if m else {"prefix": "?", "level": "?", "tags": name}


@torch.no_grad()
def fused_points(model, cams, masks, ras, max_points=300_000, seed=0):
    pts = []
    for cam, m in zip(cams, masks):
        z = (model.means.detach() @ cam.R.T + cam.t)[:, 2:3]
        out = model.render(ras, cam, torch.zeros(1, device=z.device), features=z)
        a = out["alpha"]
        ok = (a > 0.5) & m
        if not bool(ok.any()):
            continue
        D = out["image"][..., 0] / a.clamp_min(1e-6)
        v, u = torch.nonzero(ok, as_tuple=True)
        d = D[v, u]
        x = ((u.float() + 0.5) - cam.cx) / cam.fx * d
        y = ((v.float() + 0.5) - cam.cy) / cam.fy * d
        Xc = torch.stack([x, y, d], -1)
        pts.append((Xc - cam.t) @ cam.R)          # camera -> world: R^T (Xc - t)
    P = torch.cat(pts) if pts else torch.zeros(0, 3, device=model.means.device)
    if len(P) > max_points:
        P = P[torch.randperm(len(P), generator=torch.Generator().manual_seed(seed))[:max_points].to(P.device)]
    return P


@torch.no_grad()
def geometry(model, cams, masks, ref, ras, taus=(0.01, 0.02)):
    P = fused_points(model, cams, masks, ras)
    if len(P) == 0:
        return {"geo_acc": np.nan, "geo_comp": np.nan, "chamfer": np.nan, "n_fused": 0, **{f"fscore_{int(t * 1000):03d}": 0.0 for t in taus}}
    d_pr = nearest_dist(P, ref)
    d_rp = nearest_dist(ref, P)
    out = {"geo_acc": float(d_pr.mean()), "geo_comp": float(d_rp.mean()), "n_fused": len(P)}
    out["chamfer"] = 0.5 * (out["geo_acc"] + out["geo_comp"])
    for t in taus:
        p, r = float((d_pr < t).float().mean()), float((d_rp < t).float().mean())
        out[f"fscore_{int(t * 1000):03d}"] = 2 * p * r / max(p + r, 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ras, bg = get_rasterizer("torch"), torch.ones(3, device=dev)
    rows, cache = [], {}
    for rd in map(Path, args.run_dirs):
        cfg = yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))
        sc, tc, ic = cfg["scene"], cfg.get("train", {}), cfg.get("init", {})
        key = (sc["root"], sc["n_train_views"])
        if key not in cache:
            views, test_cams, ref = load_eval_scene(cfg, dev, "strict", 0.8, 16)
            cams, masks, _ = load_dtu_views(sc["root"], sc.get("downscale", 4))
            tr_i, te_i = split_eval(cams, sc["n_train_views"], 16, 16)
            gcams = [cams[i].to(dev) for i in list(tr_i) + list(te_i)]
            gmasks = [torch.from_numpy(masks[i]).to(dev) for i in list(tr_i) + list(te_i)]
            cache = {key: (views, test_cams, ref, gcams, gmasks)}
        views, test_cams, ref, gcams, gmasks = cache[key]
        photo = sc.get("photometric") or {}
        info = dict(run=rd.name, scan=Path(sc["root"]).name, N=sc["n_train_views"], **parse(rd.name),
                    appearance=tc.get("appearance", "none"), init=ic.get("type", "random"),
                    renderer=cfg.get("renderer", "torch"), sh_degree=tc.get("sh_degree", 0),
                    interv=tc.get("interv", "none"), interv_target=tc.get("interv_target", "off"),
                    freeze="+".join(tc.get("freeze", []) or []), topo_freeze_from=tc.get("topo_freeze_from", -1),
                    parent=ic.get("ckpt", ""), exposure_stops=photo.get("exposure_stops", 0.0))
        for seed in args.seeds:
            ck = rd / f"seed_{seed}" / "base.pt"
            if not ck.exists():
                continue
            model = Trainer.load(ck, TrainConfig(**tc), ras, bg, dev).model
            row = dict(info, seed=seed)
            row.update(metrics(model, views, test_cams, ref, ras, bg, 0.03, (0.05, 0.1)))
            cb = fm2.contrib(model, test_cams, ras, bg)
            row["wsig_01"] = float((cb * fm2.sigma_far(model, ref)).sum() / cb.sum().clamp_min(1e-12))
            row["psnr_aligned"] = fm2.aligned_psnr(model, test_cams, ras, bg)
            row.update(geometry(model, gcams, gmasks, ref, ras))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                   if k in ("run", "seed", "wfloat_01", "chamfer", "fscore_010", "psnr")}, flush=True)
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
