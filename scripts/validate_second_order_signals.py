"""W4 diagnostics: sample/damping stability and optional generated-scene GPU profiling.

By default this uses constructed, UNTRAINED Gaussian fixtures. --run_dir selects a verified
trained checkpoint instead. Ranking stability is not failure association or faithfulness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.camera import orbit_cameras
from frecon.gaussians import GaussianModel, inverse_sigmoid
from frecon.render import get_rasterizer
from frecon.signals.second_order_export import NAMES, compute
from frecon.signals import fisher, pup_sensitivity, s_geom
from frecon.signals.common import SignalContext
from scripts.export_second_order import load_checkpoint
from frecon.data import load_scene
from tests.test_second_order import tiny_context


def targets(model, cams, rasterize, bg):
    with torch.no_grad():
        for cam in cams:
            cam.image = (model.render(rasterize, cam, bg)["image"] * 0.95).clamp(0, 1)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sweep(out, ctx=None, title="Constructed-scene"):
    if ctx is None:
        ctx = tiny_context()
        targets(ctx.model, ctx.cams, ctx.rasterize, ctx.bg)
    reference, _ = compute(ctx.model, ctx.cams, ctx.rasterize, ctx.bg, n_samples=256)
    reference = {name: value.cpu().numpy() for name, value in reference.items()}
    rows = []
    for seed in range(3):
        for samples in (4, 16, 64):
            # Cache raw blocks across epsilon choices; the checkpoint/cameras stay immutable.
            trial = SignalContext(ctx.model, ctx.cams, ctx.rasterize, ctx.bg, seed)
            for eps in (1e-8, 1e-6, 1e-4):
                scores = dict(fisher=fisher.compute(trial, samples),
                              pup=pup_sensitivity.compute(trial, samples, eps),
                              curvature=s_geom.curvature(trial, samples, eps),
                              gradient_inconsistency=s_geom.inconsistency(trial),
                              s_geom=s_geom.s_geom(trial, samples, eps),
                              s_geom_ranksum=s_geom.s_geom_ranksum(trial, samples, eps))
                for name, value in scores.items():
                    value = value.cpu().numpy()
                    rho = float(spearmanr(value, reference[name]).statistic) if np.unique(value).size > 1 else float("nan")
                    k = max(1, len(value) // 5)
                    overlap = len(set(np.argsort(value, kind="stable")[-k:]) &
                                  set(np.argsort(reference[name], kind="stable")[-k:])) / k
                    rows.append(dict(probe_seed=seed, n_samples=samples, eps=eps, signal=name,
                                     min=float(value.min()), median=float(np.median(value)), max=float(value.max()),
                                     unique_scores=len(np.unique(value)), spearman_reference=rho,
                                     top20_overlap=overlap))
            print(f"stability: seed={seed}, S={samples} complete", flush=True)
    write_csv(out / "stability.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, name in zip(axes.flat, NAMES):
        for eps in (1e-8, 1e-6, 1e-4):
            groups = [np.array([r["spearman_reference"] for r in rows if r["signal"] == name
                                and r["eps"] == eps and r["n_samples"] == s]) for s in (4, 16, 64)]
            ax.errorbar([4, 16, 64], [g.mean() for g in groups], yerr=[g.std(ddof=1) for g in groups],
                        marker="o", capsize=2, label=f"eps={eps:g}")
        ax.set(xscale="log", ylim=(-0.05, 1.05), title=name, xlabel="Probes per view", ylabel="Spearman vs S=256")
        ax.legend(fontsize=7)
    fig.suptitle(f"{title} ranking stability; 3 probe seeds; reference eps=1e-8")
    fig.tight_layout()
    fig.savefig(out / "stability.png", dpi=160)
    plt.close(fig)


def profile(out, n, samples):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    generator = torch.Generator().manual_seed(123)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1
    model = GaussianModel(
        (torch.rand(n, 3, generator=generator) - 0.5) * 1.4,
        (0.008 + torch.rand(n, 3, generator=generator) * 0.007).log(), quats,
        inverse_sigmoid(torch.full((n,), 0.5)),
        inverse_sigmoid(0.1 + 0.8 * torch.rand(n, 3, generator=generator)),
    )
    model = GaussianModel.from_state_dict(model.state_dict(), "cuda")
    cams = [c.to("cuda") for c in orbit_cameras(8, 4.0, 160, 160, 30)]
    rasterize, bg = get_rasterizer("torch"), torch.ones(3, device="cuda")
    targets(model, cams, rasterize, bg)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    scores, options = compute(model, cams, rasterize, bg, n_samples=samples)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    rows = [dict(signal=name, n_gaussians=n, views=8, resolution=160, n_samples=samples,
                 bundle_seconds=seconds, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(), min=float(value.min()),
                 max=float(value.max()), gpu=torch.cuda.get_device_name()) for name, value in scores.items()]
    write_csv(out / "profile.csv", rows)
    (out / "profile_options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    print(f"profile: all {len(scores)} scores, N={n}, {seconds:.2f}s, "
          f"peak allocated={torch.cuda.max_memory_allocated()/2**20:.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="results/second_order/w4")
    parser.add_argument("--profile_n", type=int, default=0)
    parser.add_argument("--profile_samples", type=int, default=4)
    parser.add_argument("--run_dir", help="Use a trained D1 checkpoint for the stability sweep")
    parser.add_argument("--checkpoint", help="Relocated checkpoint, still validated against its manifest")
    parser.add_argument("--config", default="results/phase0/config_used.yaml")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint_meta = None
    if args.run_dir:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, checkpoint_meta = load_checkpoint(args.run_dir, device, args.checkpoint)
        rasterize, bg = get_rasterizer(cfg.get("renderer", "torch")), torch.ones(3, device=device)
        train, _, _, _ = load_scene(cfg["scene"], rasterize, bg, device)
        sweep(out, SignalContext(model, train, rasterize, bg), "Trained seed-" + str(checkpoint_meta["seed"]))
    else:
        sweep(out)
    if args.profile_n:
        profile(out, args.profile_n, args.profile_samples)
    source_paths = list(Path("frecon/signals").glob("*.py")) + [Path(__file__)]
    if checkpoint_meta:
        source_paths.append(Path(args.config))
    metadata = dict(python=sys.version, torch=torch.__version__,
                    fixture="trained_checkpoint" if checkpoint_meta else "constructed_untrained",
                    scene_seed=cfg["scene"].get("seed", 0) if checkpoint_meta else 123,
                    checkpoint=checkpoint_meta, command=sys.argv,
                    reference_samples=256, samples=[4, 16, 64], eps=[1e-8, 1e-6, 1e-4],
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    dirty_state=subprocess.check_output(["git", "status", "--short"], text=True),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
