"""Phase 0: D2 proof of concept.

For each seed:
  1. train a base 3DGS model on the sparse training views;
  2. compute reliability signals on it and select the top-K (least reliable) Gaussians,
     plus `n_random` size-matched random selections;
  3. remove each selection and re-optimize for `reopt_steps` (densification off);
     also run two null interventions (no removal) with different view orders;
  4. measure, against null_a: image-space locality (D2.1), held-out error and local
     geometry change (D2.2), and the same quantities for null_b vs null_a (noise floor).

Every stage is cached under out_dir/seed_<s>/, so the script can be interrupted and
resumed. Aggregation and the go/no-go decision are in scripts/analyze_phase0.py.

Usage:  python scripts/phase0.py --config configs/phase0.yaml [--seeds 0] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.camera import scene_extent  # noqa: E402
from frecon.data import load_scene  # noqa: E402
from frecon.gaussians import GaussianModel  # noqa: E402
from frecon.measure import (auroc, displacement_split, floater_labels, heldout_errors, image_delta,  # noqa: E402
                            local_geometry, locality_ratio, neighborhood, region_masks, render_views)
from frecon.render import get_rasterizer  # noqa: E402
from frecon.signals import compute_signals, top_fraction  # noqa: E402
from frecon.train import TrainConfig, Trainer, set_seed  # noqa: E402

QUICK = {"train": {"iterations": 300, "densify_from": 100, "densify_until": 200, "densify_every": 50,
                   "opacity_reset_every": 10_000, "log_every": 100},
         "intervention": {"reopt_steps": 50, "n_random": 1}}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deep_update(d, u):
    for k, v in u.items():
        d[k] = deep_update(d.get(k, {}), v) if isinstance(v, dict) else v
    return d


def train_base(cfg, seed, train_cams, rasterize, bg, dev, path: Path):
    tcfg = TrainConfig(**cfg["train"])
    if path.exists():
        return Trainer.load(path, tcfg, rasterize, bg, dev)
    set_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model = GaussianModel.random_in_box(cfg["init"]["n_points"], cfg["init"]["half_extent"], gen, dev)
    tr = Trainer(model, tcfg, scene_extent(train_cams), rasterize, bg)
    log(f"seed {seed}: training base model ({tcfg.iterations} iters, {len(train_cams)} views)")
    t0 = time.time()
    tr.fit(train_cams, seed=seed, log=lambda m: log(m))
    log(f"seed {seed}: base done in {(time.time() - t0) / 60:.1f} min, {tr.model.n} Gaussians")
    tr.save(path)
    return tr


def reoptimize(cfg, base_path, remove, steps, seed, train_cams, test_cams, rasterize, bg, dev):
    tr = Trainer.load(base_path, TrainConfig(**cfg["train"]), rasterize, bg, dev)
    if remove is not None:
        tr.prune(~remove)
    set_seed(seed)
    tr.fit(train_cams, iterations=steps, densify=False, seed=seed, means_lr=tr.final_means_lr(), log=None)
    m = tr.model
    return {"means": m.means.detach().cpu(), "opac": m.opacities.detach().cpu(),
            "renders": render_views(m, test_cams, rasterize, bg).half().cpu()}


def cached(path: Path, fn):
    if path.exists():
        return torch.load(path, weights_only=False)
    out = fn()
    torch.save(out, path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # small GPUs: release the previous run's blocks
    return out


def run_seed(cfg, seed, scene, rasterize, bg, dev):
    train_cams, test_cams, ref_pts, _ = scene
    sdir = Path(cfg["out_dir"]) / f"seed_{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    icfg = cfg["intervention"]

    base_path = sdir / "base.pt"
    base_tr = train_base(cfg, seed, train_cams, rasterize, bg, dev, base_path)
    base = base_tr.model
    del base_tr  # the optimizer state is not needed here; re-optimization reloads it from disk
    base_means = base.means.detach().clone()

    # ---- signals and selections ------------------------------------------------------------
    sig = cached(sdir / "signals.pt", lambda: {k: v.cpu() for k, v in compute_signals(
        base, train_cams, rasterize, bg, names=("random",) + tuple(icfg["signals"]), seed=seed).items()})
    sig = {k: v.to(dev) for k, v in sig.items()}
    selections = {}
    for name in icfg["signals"]:
        selections[name] = top_fraction(sig[name], icfg["frac"])
    gen = torch.Generator(device=dev).manual_seed(10_000 + seed)
    for r in range(icfg["n_random"]):
        selections[f"random_{r}"] = top_fraction(torch.rand(base.n, generator=gen, device=dev), icfg["frac"])

    # ---- re-optimization runs ------------------------------------------------------------------
    seed_a, seed_b = 1000 + seed, 2000 + seed
    steps = icfg["reopt_steps"]

    def run(name, remove, rseed):
        path = sdir / f"run_{name}.pt"
        if not path.exists():
            log(f"seed {seed}: re-optimizing '{name}' ({steps} steps)")
        out = cached(path, lambda: reoptimize(cfg, base_path, remove, steps, rseed, train_cams, test_cams,
                                              rasterize, bg, dev))
        return {k: v.to(dev) for k, v in out.items()}

    null_a = run("null_a", None, seed_a)
    null_b = run("null_b", None, seed_b)
    runs = {name: run(name, sel, seed_a) for name, sel in selections.items()}

    # ---- measurements -------------------------------------------------------------------------
    base_renders = render_views(base, test_cams, rasterize, bg)
    ra, rb = null_a["renders"].float(), null_b["renders"].float()
    err_base = heldout_errors(base_renders, test_cams)
    err_a, err_b = heldout_errors(ra, test_cams), heldout_errors(rb, test_cams)

    with torch.no_grad():
        heldout_contrib = torch.zeros(base.n, device=dev)
        for c in test_cams:
            heldout_contrib += base.render(rasterize, c, bg, return_contrib=True)["contrib"]
    floaters = None
    if ref_pts is not None:
        floaters = floater_labels(base, ref_pts, heldout_contrib, cfg["labels"]["tau"])

    radius = icfg["neighborhood_radius"]
    pts_a = null_a["means"][null_a["opac"] > 0.1]
    pts_b = null_b["means"][null_b["opac"] > 0.1]

    results = {"seed": seed, "n_gaussians": base.n, "err_base": err_base, "err_null_a": err_a,
               "err_null_b": err_b, "selections": {}}
    if floaters is not None:
        results["floater_fraction"] = float(floaters.float().mean())
        results["n_floaters"] = int(floaters.sum())

    for name, sel in selections.items():
        rec = {"kind": "random" if name.startswith("random") else "signal", "n_selected": int(sel.sum())}
        masks = region_masks(base, sel, test_cams, rasterize, bg, icfg["region_alpha_thresh"])
        ri = runs[name]["renders"].float()

        # D2.1 image-space locality, and the same split for the noise floor
        d_in, d_out, n_in, n_out = image_delta(ri, ra, masks)
        nd_in, nd_out, _, _ = image_delta(rb, ra, masks)
        rec["image"] = {"d_in": d_in, "d_out": d_out, "LR": locality_ratio(d_in, d_out),
                        "noise_d_in": nd_in, "noise_d_out": nd_out,
                        "snr_in": d_in / max(nd_in, 1e-12), "snr_out": d_out / max(nd_out, 1e-12),
                        "region_frac": n_in / max(n_in + n_out, 1)}

        # D2.1 in 3D: displacement of surviving Gaussians near vs far from the removed ones
        keep = ~sel
        near = neighborhood(base_means[keep], base_means[sel], radius)
        disp_in, disp_out = displacement_split(runs[name]["means"], null_a["means"][keep], near)
        ndisp_in, ndisp_out = displacement_split(null_b["means"][keep], null_a["means"][keep], near)
        rec["param"] = {"disp_in": disp_in, "disp_out": disp_out, "LR": locality_ratio(disp_in, disp_out),
                        "noise_disp_in": ndisp_in, "noise_disp_out": ndisp_out,
                        "snr_in": disp_in / max(ndisp_in, 1e-12)}

        # D2.2 direction: held-out error and local geometry change relative to null_a
        err_i = heldout_errors(ri, test_cams)
        rec["heldout"] = {k: err_i[k] - err_a[k] for k in err_i}
        rec["heldout_noise"] = {k: err_b[k] - err_a[k] for k in err_b}
        if ref_pts is not None:
            pts_i = runs[name]["means"][runs[name]["opac"] > 0.1]
            centers = base_means[sel]
            gi = local_geometry(pts_i, ref_pts, centers, radius)
            ga = local_geometry(pts_a, ref_pts, centers, radius)
            gb = local_geometry(pts_b, ref_pts, centers, radius)
            rec["geometry"] = {"int": gi, "null_a": ga, "null_b": gb,
                               "d_cd": gi["cd"] - ga["cd"], "d_acc": gi["acc"] - ga["acc"],
                               "d_comp": gi["comp"] - ga["comp"], "noise_d_cd": gb["cd"] - ga["cd"]}

        # D1 sanity: how many selected Gaussians are floaters; AUROC of the signal
        if floaters is not None:
            rec["floater_precision"] = float(floaters[sel].float().mean())
            if rec["kind"] == "signal":
                rec["auroc_floater"] = auroc(sig[name], floaters)
        results["selections"][name] = rec

    (sdir / "metrics.json").write_text(json.dumps(results, indent=2))
    log(f"seed {seed}: metrics written to {sdir / 'metrics.json'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test settings (minutes, not hours)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg = deep_update(cfg, QUICK)
        cfg["out_dir"] = cfg["out_dir"].rstrip("/") + "_quick"
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    seeds = args.seeds if args.seeds else cfg["seeds"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    dev = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=dev)
    log(f"loading scene ({cfg['scene'].get('type', 'synthetic')}) on {dev}")
    scene = load_scene(cfg["scene"], rasterize, bg, dev)
    for seed in seeds:
        run_seed(cfg, seed, scene, rasterize, bg, dev)
    log("done. Next: python scripts/analyze_phase0.py --run_dir " + str(out))


if __name__ == "__main__":
    main()
