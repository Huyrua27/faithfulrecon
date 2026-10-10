"""Phase 0.5 — D2 protocol v2 on the Phase 0 checkpoints.

Changes vs Phase 0 (docs/phase0.md, "Known issues"):
  1. Image region mask: besides the footprint mask (selected Gaussians rendered alone), a
     WEIGHT mask from the real, occlusion-aware blending weights of the selected Gaussians in
     the base render: pixels where they carry >= `region_weight_thresh` of the pixel.
  2. 3D locality (parameter-space displacement) enters the decision rule (analyze_d2.py).
  3. Direction is split into expected vs opposite (analyze_d2.py), and the local geometry
     change is split into accuracy (floaters) and completeness (holes).
  4. Response is measured right after removal (step 0) and after `snapshots` re-optimization
     steps, from a single re-optimization run per selection.

The base checkpoints and signals are reused from `reuse_from` (copied), so selections are
identical to Phase 0. Cached and resumable like phase0.py.

Usage: python scripts/d2_v2.py --config configs/phase0_5.yaml [--seeds 0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frecon.data import load_scene  # noqa: E402
from frecon.measure import (displacement_split, floater_labels, heldout_errors, image_delta,  # noqa: E402
                            local_geometry, locality_ratio, neighborhood, region_masks, render_views)
from frecon.render import get_rasterizer  # noqa: E402
from frecon.signals import top_fraction  # noqa: E402
from frecon.train import TrainConfig, Trainer, set_seed  # noqa: E402
from phase0 import deep_update, log  # noqa: E402


@torch.no_grad()
def weight_masks(model, selected, cams, rasterize, thresh):
    """Pixels where the selected Gaussians carry >= thresh of the blending weight in `model`."""
    feats = selected.float()[:, None]
    zero = torch.zeros(1, device=feats.device)
    return torch.stack([model.render(rasterize, c, zero, features=feats)["image"][..., 0] >= thresh
                        for c in cams])


def snapshot(model, test_cams, rasterize, bg):
    return {"means": model.means.detach().cpu(), "opac": model.opacities.detach().cpu(),
            "renders": render_views(model, test_cams, rasterize, bg).half().cpu()}


def reoptimize(cfg, base_path, remove, steps_list, seed, train_cams, test_cams, rasterize, bg, dev):
    tr = Trainer.load(base_path, TrainConfig(**cfg["train"]), rasterize, bg, dev)
    if remove is not None:
        tr.prune(~remove)
    snaps = {}
    if 0 in steps_list:
        snaps[0] = snapshot(tr.model, test_cams, rasterize, bg)
    total = max(steps_list)

    def cb(it):
        if it in steps_list:
            snaps[it] = snapshot(tr.model, test_cams, rasterize, bg)

    set_seed(seed)
    if total > 0:
        tr.fit(train_cams, iterations=total, densify=False, seed=seed, means_lr=tr.final_means_lr(),
               log=None, callback=cb)
    return snaps


def to_dev(snaps, dev):
    return {t: {k: v.to(dev) for k, v in s.items()} for t, s in snaps.items()}


def run_seed(cfg, seed, scene, rasterize, bg, dev):
    train_cams, test_cams, ref_pts, _ = scene
    icfg = cfg["intervention"]
    steps_list = sorted(set(icfg["snapshots"]))
    sdir = Path(cfg["out_dir"]) / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    src = Path(cfg["reuse_from"]) / f"seed_{seed}"
    for f in ("base.pt", "signals.pt"):
        if not (sdir / f).exists():
            shutil.copy2(src / f, sdir / f)
    base_path = sdir / "base.pt"
    base = Trainer.load(base_path, TrainConfig(**cfg["train"]), rasterize, bg, dev).model
    base_means = base.means.detach().clone()

    sig = {k: v.to(dev) for k, v in torch.load(sdir / "signals.pt", weights_only=False).items()}
    selections = {name: top_fraction(sig[name], icfg["frac"]) for name in icfg["signals"]}
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)  # same random selections as Phase 0
    for r in range(icfg["n_random"]):
        selections[f"random_{r}"] = top_fraction(torch.rand(base.n, generator=gen, device=dev), icfg["frac"])

    seed_a, seed_b = 1000 + seed, 2000 + seed

    def run(name, remove, rseed):
        path = sdir / f"run_{name}.pt"
        if path.exists():
            snaps = torch.load(path, weights_only=False)
        else:
            log(f"seed {seed}: re-optimizing '{name}' (snapshots {steps_list})")
            snaps = reoptimize(cfg, base_path, remove, steps_list, rseed, train_cams, test_cams, rasterize, bg, dev)
            torch.save(snaps, path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return to_dev(snaps, dev)

    null_a = run("null_a", None, seed_a)
    null_b = run("null_b", None, seed_b)
    runs = {name: run(name, sel, seed_a) for name, sel in selections.items()}

    with torch.no_grad():
        contrib = torch.zeros(base.n, device=dev)
        for c in test_cams:
            contrib += base.render(rasterize, c, bg, return_contrib=True)["contrib"]
    floaters = floater_labels(base, ref_pts, contrib, cfg["labels"]["tau"]) if ref_pts is not None else None
    radius = icfg["neighborhood_radius"]

    results = {"seed": seed, "n_gaussians": base.n, "steps": steps_list, "selections": {}}
    for name, sel in selections.items():
        keep = ~sel
        masks = {"footprint": region_masks(base, sel, test_cams, rasterize, bg, icfg["region_alpha_thresh"]),
                 "weight": weight_masks(base, sel, test_cams, rasterize, icfg["region_weight_thresh"])}
        near = neighborhood(base_means[keep], base_means[sel], radius)
        centers = base_means[sel]
        rec = {"kind": "random" if name.startswith("random") else "signal", "n_selected": int(sel.sum()),
               "region_frac": {k: float(m.float().mean()) for k, m in masks.items()}, "by_step": {}}
        if floaters is not None:
            rec["floater_precision"] = float(floaters[sel].float().mean())
        for t in steps_list:
            ri, ra, rb = (x[t]["renders"].float() for x in (runs[name], null_a, null_b))
            st = {"image": {}, "noise": t > 0}
            for mk, m in masks.items():
                d_in, d_out, n_in, _ = image_delta(ri, ra, m)
                e = {"d_in": d_in, "d_out": d_out, "LR": locality_ratio(d_in, d_out) if n_in else float("nan")}
                if t > 0:
                    nd_in, nd_out, _, _ = image_delta(rb, ra, m)
                    e.update(noise_d_in=nd_in, noise_d_out=nd_out, snr_in=d_in / max(nd_in, 1e-12))
                st["image"][mk] = e
            di, do = displacement_split(runs[name][t]["means"], null_a[t]["means"][keep], near)
            st["param"] = {"disp_in": di, "disp_out": do, "LR": locality_ratio(di, do)}
            if t > 0:
                ndi, ndo = displacement_split(null_b[t]["means"][keep], null_a[t]["means"][keep], near)
                st["param"].update(noise_disp_in=ndi, noise_disp_out=ndo, snr_in=di / max(ndi, 1e-12))
            ei, ea, eb = (heldout_errors(r, test_cams) for r in (ri, ra, rb))
            st["heldout"] = {k: ei[k] - ea[k] for k in ei}
            st["heldout_noise"] = {k: eb[k] - ea[k] for k in eb}
            if ref_pts is not None:
                pts = {k: x[t]["means"][x[t]["opac"] > 0.1] for k, x in
                       (("int", runs[name]), ("a", null_a), ("b", null_b))}
                gi, ga, gb = (local_geometry(pts[k], ref_pts, centers, radius) for k in ("int", "a", "b"))
                st["geometry"] = {f"d_{k}": gi[k] - ga[k] for k in ("acc", "comp", "cd")}
                st["geometry"].update({f"noise_d_{k}": gb[k] - ga[k] for k in ("acc", "comp", "cd")})
            rec["by_step"][str(t)] = st
        results["selections"][name] = rec

    (sdir / "metrics.json").write_text(json.dumps(results, indent=2))
    log(f"seed {seed}: metrics written to {sdir / 'metrics.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_5.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    own = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((Path(own["reuse_from"]) / "config.yaml").read_text(encoding="utf-8"))
    cfg = deep_update(cfg, own)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    dev = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=dev)
    log(f"loading scene ({cfg['scene'].get('type', 'synthetic')}) on {dev}")
    scene = load_scene(cfg["scene"], rasterize, bg, dev)
    t0 = time.time()
    for seed in (args.seeds or cfg["seeds"]):
        run_seed(cfg, seed, scene, rasterize, bg, dev)
    log(f"done in {(time.time() - t0) / 60:.1f} min. Next: python scripts/analyze_d2.py --run_dir {out}")


if __name__ == "__main__":
    main()
