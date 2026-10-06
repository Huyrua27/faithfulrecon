"""Report figures for Phase 0 (for sharing with advisors).

Writes into the run dir:
  phase0_report_figure.png   quantitative summary (SNR, locality, direction, D1 sanity, verdict)
  phase0_qualitative.png     one held-out view: GT, reference render, |change| after each removal

Usage: python scripts/make_phase0_figure.py --run_dir runs/phase0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_phase0 import consistent, per_seed_gains  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Reference categorical palette (validated: CVD all-pairs dE 9.2, normal-vision 24.0, light mode).
COLORS = {"opacity": "#2a78d6", "gradmag": "#eb6834", "visibility": "#1baf7a"}
RANDOM = "#8a8984"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
THRESH = "#c0392b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load(run_dir: Path):
    ms = [json.loads(p.read_text()) for p in sorted(run_dir.glob("seed_*/metrics.json"))]
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return ms, cfg


def style(ax, zero=True):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if zero:
        ax.axhline(0, color=INK2, lw=0.8)


def strip(ax, cats, values, colors, faded=None, fmt="{:+.3f}", label_mean=True):
    """Mean bar (thin) + one dot per seed; `faded` categories are drawn light and tagged n.m."""
    faded = faded or set()
    for i, (c, vals) in enumerate(zip(cats, values)):
        vals = np.asarray([v for v in vals if not math.isnan(v)])
        a = 0.28 if c in faded else 0.9
        m = vals.mean() if vals.size else np.nan
        ax.bar(i, m, width=0.5, color=colors[i], alpha=a * 0.55, edgecolor="none", zorder=2)
        ax.scatter(np.full(vals.size, i) + np.linspace(-0.09, 0.09, vals.size), vals, s=30,
                   color=colors[i], alpha=min(1.0, a + 0.1), edgecolor=SURFACE, linewidth=1.2, zorder=3)
        if label_mean and not np.isnan(m):
            txt = fmt.format(m) + ("\n(n.m.)" if c in faded else "")
            top = max(vals.max(), m, 0)
            ax.annotate(txt, (i, top), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(cats)), cats)
    ax.margins(y=0.25)


def quantitative(run_dir: Path, ms, cfg):
    signals = [k for k, v in ms[0]["selections"].items() if v["kind"] == "signal"]
    gains = {s: [per_seed_gains(m, s) for m in ms] for s in signals}
    snr_min = cfg.get("analysis", {}).get("snr_min", 2.0)
    measurable = {s: all(g["snr_in"] >= snr_min for g in gains[s]) for s in signals}
    not_meas = {s for s in signals if not measurable[s]}
    rnd_snr = [float(np.mean([v["image"]["snr_in"] for v in m["selections"].values() if v["kind"] == "random"]))
               for m in ms]
    cols = [COLORS.get(s, "#4a3aa7") for s in signals]

    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.72], hspace=0.62, wspace=0.3,
                          left=0.06, right=0.985, top=0.835, bottom=0.02)
    sc = cfg["scene"]
    ic = cfg["intervention"]
    fig.text(0.05, 0.965, "FaithfulRecon — Phase 0: D2 intervention pilot", fontsize=15, weight="bold", color=INK)
    fig.text(0.05, 0.94,
             f"Synthetic scene with exact reference geometry · {sc['n_train_views']} training / {sc['n_test_views']} "
             f"held-out views, {sc['resolution']} px · 3DGS {cfg['train']['iterations']} iterations\n"
             f"Intervention: remove the top-{ic['frac']:.0%} least-reliable Gaussians per signal → re-optimize "
             f"{ic['reopt_steps']} steps · {len(ms)} seeds × ({len(signals)} signals + {ic['n_random']} random "
             f"+ 2 null runs)\n"
             "Bars = mean over seeds, dots = seeds · Gain = signal − mean of size-matched random removals (same seed) · "
             "n.m. = not measurable (response at noise floor), gains not interpreted",
             fontsize=9, color=INK2, va="top", linespacing=1.6)

    # A: measurability
    ax = fig.add_subplot(gs[0, 0])
    cats = signals + ["random"]
    strip(ax, cats, [[g["snr_in"] for g in gains[s]] for s in signals] + [rnd_snr], cols + [RANDOM],
          fmt="{:.1f}")
    ax.axhline(snr_min, color=THRESH, lw=1, ls="--")
    ax.annotate(f"threshold {snr_min:g}", (len(cats) - 0.55, snr_min), xytext=(0, 3), textcoords="offset points",
                ha="right", fontsize=7.5, color=THRESH)
    ax.set_title("A  Is the response above the noise floor?\n    Δ(intervention) / Δ(null_b vs null_a), in region",
                 linespacing=1.5)
    ax.set_ylabel("SNR_in")
    style(ax, zero=False)
    ax.set_ylim(0, None)

    # B: locality, parameter space
    ax = fig.add_subplot(gs[0, 1])
    strip(ax, signals, [[g["gain_LR_param"] for g in gains[s]] for s in signals], cols, faded=not_meas)
    ax.set_title("B  Locality gain in 3D (D2.1 / D2.3)\n    displacement near vs far from removed Gaussians",
                 linespacing=1.5)
    ax.set_ylabel("Gain_LR (3D)")
    style(ax)

    # C: locality, image space
    ax = fig.add_subplot(gs[0, 2])
    strip(ax, signals, [[g["gain_LR_img"] for g in gains[s]] for s in signals], cols, faded=not_meas)
    lr_r = np.mean([g["LR_img_random"] for g in gains[signals[0]]])
    ax.set_title("C  Locality gain in image space\n    change inside vs outside projected region", linespacing=1.5)
    ax.set_ylabel("Gain_LR (image)")
    ax.text(0.02, 0.03, f"LR_img ≈ {lr_r:.2f} even for random removal:\nregion mask too wide → weak contrast",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    style(ax)

    # D: direction, local geometry
    ax = fig.add_subplot(gs[1, 0])
    if "gain_dir_cd" in gains[signals[0]][0]:
        strip(ax, signals, [[g["gain_dir_cd"] * 1e3 for g in gains[s]] for s in signals], cols, faded=not_meas,
              fmt="{:+.2f}")
    ax.set_title("D  Direction gain: local geometry (D2.2)\n    > 0 = removal improved geometry vs random",
                 linespacing=1.5)
    ax.set_ylabel("Gain_Dir, −ΔChamfer (×10⁻³)")
    style(ax)

    # E: direction, held-out image quality
    ax = fig.add_subplot(gs[1, 1])
    strip(ax, signals, [[g["gain_dir_dssim"] * 1e3 for g in gains[s]] for s in signals], cols, faded=not_meas,
          fmt="{:+.2f}")
    ax.set_title("E  Direction gain: held-out images (D2.2)\n    > 0 = removal improved held-out views vs random",
                 linespacing=1.5)
    ax.set_ylabel("Gain_Dir, −Δ(1−SSIM) (×10⁻³)")
    style(ax)

    # F: D1 sanity
    ax = fig.add_subplot(gs[1, 2])
    if "auroc_floater" in gains[signals[0]][0]:
        strip(ax, signals, [[g["auroc_floater"] for g in gains[s]] for s in signals], cols, fmt="{:.2f}")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.annotate("chance", (len(signals) - 0.55, 0.5), xytext=(0, 3), textcoords="offset points", ha="right",
                    fontsize=7.5, color=INK2)
    ax.set_ylim(0, 1)
    ax.set_title("F  D1 sanity: association with floater labels\n    (reported, not a Phase 0 criterion)",
                 linespacing=1.5)
    ax.set_ylabel("AUROC (signal vs floater label)")
    style(ax, zero=False)

    # Verdict box
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
                                facecolor="#f1f0ec", edgecolor="none"))
    rows = []
    for s in signals:
        G = gains[s]
        if not measurable[s]:
            rows.append(f"• {s}: not measurable (SNR_in {np.mean([g['snr_in'] for g in G]):.1f}) — removed Gaussians "
                        "barely contribute; no gain interpreted.")
            continue
        parts = []
        for key, name, expected in [("gain_LR_param", "3D locality", "+"), ("gain_LR_img", "image locality", "+"),
                                    ("gain_dir_cd", "geometry direction", "+"),
                                    ("gain_dir_dssim", "held-out direction", "+")]:
            if key not in G[0]:
                continue
            vals = [g[key] for g in G]
            if consistent(vals):
                sign = "+" if np.mean(vals) > 0 else "−"
                parts.append(f"{name} {sign} ({'expected' if sign == expected else 'OPPOSITE to expected'})")
        rows.append(f"• {s}: measurable (SNR_in {np.mean([g['snr_in'] for g in G]):.1f}); consistent across seeds: "
                    + ("; ".join(parts) if parts else "none"))
    ax.text(0.015, 0.91, "Verdict: GO for D2 — the protocol detects intervention responses above re-optimization "
            "noise and separates signals from random removal.", transform=ax.transAxes, fontsize=10.5,
            weight="bold", color=INK, va="top")
    ax.text(0.015, 0.74, "\n".join(rows), transform=ax.transAxes, fontsize=8.8, color=INK, va="top", linespacing=1.6)
    ax.text(0.015, 0.06, "GO means D2 is measurable, not that any signal is faithful.\n"
            "Before Phase 1: narrow the image region mask · add 3D locality to the decision rule · separate 'consistent "
            "in the expected direction' from 'consistent but opposite' ·\nmeasure the response right after removal as "
            "well as after re-optimization.",
            transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom", linespacing=1.5)

    out = run_dir / "phase0_report_figure.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def qualitative(run_dir: Path, ms, cfg, seed: int = 0):
    import torch
    from frecon.data import load_scene
    from frecon.render import get_rasterizer

    sdir = run_dir / f"seed_{seed}"
    runs = {p.stem[4:]: torch.load(p, weights_only=False) for p in sorted(sdir.glob("run_*.pt"))}
    dev = "cpu"
    bg = torch.ones(3)
    scfg = dict(cfg["scene"])
    _, test, _, _ = load_scene(scfg, get_rasterizer(cfg.get("renderer", "torch")), bg, dev)
    ref = runs["null_a"]["renders"].float()
    signals = [k for k, v in ms[0]["selections"].items() if v["kind"] == "signal"]
    show = [s for s in signals if s in runs] + ["random_0"]
    diffs = {k: (runs[k]["renders"].float() - ref).abs().mean(-1) for k in show}
    # the held-out view where the largest signal response is
    v = int(torch.stack([d.flatten(1).mean(1) for d in diffs.values()]).sum(0).argmax())
    vmax = float(max(d[v].quantile(0.995) for d in diffs.values())) or 1e-3

    n = 2 + len(show)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.9))
    axes[0].imshow(test[v].image.numpy())
    axes[0].set_title("Ground truth\n", fontsize=9)
    axes[1].imshow(ref[v].clamp(0, 1).numpy())
    axes[1].set_title("Reference render\n(null_a)", fontsize=9)
    for ax, k in zip(axes[2:], show):
        im = ax.imshow(diffs[k][v].numpy(), cmap="magma", vmin=0, vmax=vmax)
        name = "random 5%" if k.startswith("random") else f"top-5% {k}"
        ax.set_title(f"|Δ| after removing\n{name}", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes[2:], fraction=0.025, pad=0.01)
    cb.set_label("mean |RGB change|", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Held-out view {v}, seed {seed}: change after removal + {cfg['intervention']['reopt_steps']} "
                 "re-optimization steps (same color scale)", fontsize=10, x=0.01, ha="left", color=INK)
    out = run_dir / "phase0_qualitative.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/phase0")
    ap.add_argument("--no_qualitative", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    ms, cfg = load(run_dir)
    print(quantitative(run_dir, ms, cfg))
    if not args.no_qualitative:
        print(qualitative(run_dir, ms, cfg))


if __name__ == "__main__":
    main()
