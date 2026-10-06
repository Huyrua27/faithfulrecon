"""Aggregate Phase 0 metrics over seeds, compute matched-contrast gains (D2.3), and apply
the go/no-go rule for D2. Writes report.md and phase0_summary.png into the run dir.

Go/no-go rule (a pilot heuristic, stated in advance; with 3 seeds no formal test has power):
  measurable   : the removal response is separable from re-optimization noise, i.e.
                 image SNR_in = d_in(int vs null_a) / d_in(null_b vs null_a) >= snr_min
                 in every seed. Checked for the random control too: if random removal
                 is not measurable, K or the re-optimization length must change.
  consistent   : Gain_LR or Gain_Dir has the same sign in every seed and
                 |mean| >= 2 * std / sqrt(n_seeds).
  GO for D2    : at least one signal is measurable AND consistent.

Usage: python scripts/analyze_phase0.py --run_dir runs/phase0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


def load(run_dir: Path):
    out = []
    for p in sorted(run_dir.glob("seed_*/metrics.json")):
        out.append(json.loads(p.read_text()))
    return out


def per_seed_gains(m: dict, signal: str):
    """Paired differences vs the mean of the random selections of the same seed."""
    sels = m["selections"]
    rnd = [v for k, v in sels.items() if v["kind"] == "random"]
    s = sels[signal]

    def mean(f):
        vals = [f(r) for r in rnd]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    g = {
        "gain_LR_img": s["image"]["LR"] - mean(lambda r: r["image"]["LR"]),
        "gain_LR_param": s["param"]["LR"] - mean(lambda r: r["param"]["LR"]),
        # Dir = -(change in held-out error); higher = moved in the expected direction
        "gain_dir_dssim": -(s["heldout"]["dssim"] - mean(lambda r: r["heldout"]["dssim"])),
        "gain_dir_l1": -(s["heldout"]["l1"] - mean(lambda r: r["heldout"]["l1"])),
        "snr_in": s["image"]["snr_in"],
        "snr_in_random": mean(lambda r: r["image"]["snr_in"]),
        "LR_img": s["image"]["LR"],
        "LR_img_random": mean(lambda r: r["image"]["LR"]),
        "dir_dssim": -s["heldout"]["dssim"],
        "dir_dssim_noise": abs(s["heldout_noise"]["dssim"]),
    }
    if "geometry" in s:
        g["gain_dir_cd"] = -(s["geometry"]["d_cd"] - mean(lambda r: r["geometry"]["d_cd"]))
        g["dir_cd"] = -s["geometry"]["d_cd"]
        g["dir_cd_noise"] = abs(s["geometry"]["noise_d_cd"])
    if "auroc_floater" in s:
        g["auroc_floater"] = s["auroc_floater"]
        g["floater_precision"] = s["floater_precision"]
        g["floater_precision_random"] = mean(lambda r: r.get("floater_precision"))
    return g


def consistent(vals):
    v = np.asarray([x for x in vals if not math.isnan(x)])
    if v.size < 2:
        return False
    same_sign = np.all(v > 0) or np.all(v < 0)
    return bool(same_sign and abs(v.mean()) >= 2 * v.std(ddof=1) / math.sqrt(v.size))


def fmt(vals, d=4):
    v = np.asarray([x for x in vals if not math.isnan(x)])
    if v.size == 0:
        return "n/a"
    if v.size == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/phase0")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    snr_min = cfg.get("analysis", {}).get("snr_min", 2.0)
    ms = load(run_dir)
    if not ms:
        raise SystemExit(f"no metrics found under {run_dir}")
    signals = [k for k, v in ms[0]["selections"].items() if v["kind"] == "signal"]
    seeds = [m["seed"] for m in ms]

    gains = {s: [per_seed_gains(m, s) for m in ms] for s in signals}
    rnd_snr = [np.mean([v["image"]["snr_in"] for v in m["selections"].values() if v["kind"] == "random"]) for m in ms]

    lines = ["# Phase 0 report — D2 proof of concept", ""]
    lines += [f"- Run dir: `{run_dir}`",
              f"- Seeds: {seeds}",
              f"- Scene: {cfg['scene'].get('type')} ({cfg['scene'].get('n_train_views')} training views, "
              f"{cfg['scene'].get('n_test_views')} held-out views, {cfg['scene'].get('resolution')} px)",
              f"- K = {cfg['intervention']['frac']:.0%}, removal, re-optimization {cfg['intervention']['reopt_steps']} steps, "
              f"{cfg['intervention']['n_random']} random selections per seed",
              ""]
    lines += ["## Base models", "", "| seed | #Gaussians | held-out PSNR | floater fraction |", "|---|---|---|---|"]
    for m in ms:
        ff = m.get("floater_fraction")
        lines.append(f"| {m['seed']} | {m['n_gaussians']} | {m['err_base']['psnr']:.2f} | "
                     f"{'n/a' if ff is None else f'{ff:.3%}'} |")
    lines += ["", "Noise floor (null_b vs null_a) held-out ΔPSNR per seed: " +
              ", ".join(f"{m['err_null_b']['psnr'] - m['err_null_a']['psnr']:+.3f}" for m in ms), ""]

    lines += ["## D2.1 / D2.3 — locality and matched contrast", "",
              "Values: mean ± std over seeds. Gains are paired differences vs the mean of the random selections "
              "of the same seed. SNR_in = response inside the selected region / noise floor inside the same region.", "",
              "| signal | SNR_in | SNR_in (random) | LR_img | LR_img (random) | Gain_LR (image) | Gain_LR (param) |",
              "|---|---|---|---|---|---|---|"]
    for s in signals:
        G = gains[s]
        lines.append(f"| {s} | {fmt([g['snr_in'] for g in G], 2)} | {fmt([g['snr_in_random'] for g in G], 2)} | "
                     f"{fmt([g['LR_img'] for g in G])} | {fmt([g['LR_img_random'] for g in G])} | "
                     f"{fmt([g['gain_LR_img'] for g in G])} | {fmt([g['gain_LR_param'] for g in G])} |")

    lines += ["", "## D2.2 / D2.3 — direction", "",
              "Dir = −(change in error after removal); positive = removal made the reconstruction better "
              "(or hurt it less). Gain_Dir = Dir(signal) − Dir(random).", "",
              "| signal | Dir (1−SSIM) | |noise| (1−SSIM) | Gain_Dir (1−SSIM) | Gain_Dir (L1) | Dir (local CD) | |noise| (local CD) | Gain_Dir (local CD) |",
              "|---|---|---|---|---|---|---|---|"]
    for s in signals:
        G = gains[s]
        cd = "gain_dir_cd" in G[0]
        lines.append(f"| {s} | {fmt([g['dir_dssim'] for g in G], 5)} | {fmt([g['dir_dssim_noise'] for g in G], 5)} | "
                     f"{fmt([g['gain_dir_dssim'] for g in G], 5)} | {fmt([g['gain_dir_l1'] for g in G], 5)} | "
                     f"{fmt([g['dir_cd'] for g in G], 5) if cd else 'n/a'} | "
                     f"{fmt([g['dir_cd_noise'] for g in G], 5) if cd else 'n/a'} | "
                     f"{fmt([g['gain_dir_cd'] for g in G], 5) if cd else 'n/a'} |")

    if "auroc_floater" in gains[signals[0]][0]:
        lines += ["", "## D1 sanity — floater labels (not a Phase 0 criterion)", "",
                  "| signal | AUROC vs floater | floater precision@K | precision@K (random) |", "|---|---|---|---|"]
        for s in signals:
            G = gains[s]
            lines.append(f"| {s} | {fmt([g['auroc_floater'] for g in G], 3)} | "
                         f"{fmt([g['floater_precision'] for g in G], 3)} | {fmt([g['floater_precision_random'] for g in G], 3)} |")

    # ---- decision ------------------------------------------------------------------------
    lines += ["", "## Go / no-go for D2", "",
              f"Rule: measurable = SNR_in ≥ {snr_min} in every seed; consistent = Gain has the same sign in every "
              "seed and |mean| ≥ 2·SE. GO if at least one signal is measurable and consistent.", "",
              "| signal | measurable | consistent Gain_LR (image) | consistent Gain_Dir (1−SSIM) | consistent Gain_Dir (local CD) |",
              "|---|---|---|---|---|"]
    go = False
    for s in signals:
        G = gains[s]
        meas = all(g["snr_in"] >= snr_min for g in G)
        c_lr = consistent([g["gain_LR_img"] for g in G])
        c_dir = consistent([g["gain_dir_dssim"] for g in G])
        c_cd = consistent([g["gain_dir_cd"] for g in G]) if "gain_dir_cd" in G[0] else False
        go |= meas and (c_lr or c_dir or c_cd)
        if meas:
            lines.append(f"| {s} | yes | {'yes' if c_lr else 'no'} | {'yes' if c_dir else 'no'} | "
                         f"{'yes' if c_cd else 'no'} |")
        else:
            # A response at the noise floor makes LR and Dir meaningless (removing Gaussians that
            # do not contribute trivially "beats" random removal), so gains are not interpreted.
            lines.append(f"| {s} | no | – | – | – |")
    rnd_meas = all(v >= snr_min for v in rnd_snr)
    lines += ["", f"Random control measurable (SNR_in ≥ {snr_min} in every seed): **{'yes' if rnd_meas else 'no'}** "
              f"({', '.join(f'{v:.2f}' for v in rnd_snr)})", ""]
    if len(ms) < 2:
        verdict = "INCONCLUSIVE — fewer than 2 seeds; consistency cannot be assessed."
    elif go:
        verdict = "GO — at least one signal gives a measurable and consistent D2 response."
    elif not rnd_meas:
        verdict = ("NO-GO — removal responses are not separable from re-optimization noise. "
                   "Try a larger K, fewer re-optimization steps, or a lower-noise Δ (e.g. more held-out views).")
    else:
        verdict = ("NO-GO — responses are measurable but no signal differs consistently from random. "
                   "Check masks and Δ definitions before adding signals.")
    lines += [f"**Verdict: {verdict}**", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    try:
        _plot(run_dir, signals, gains)
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")


def _plot(run_dir, signals, gains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [("snr_in", "SNR_in (image)"), ("gain_LR_img", "Gain_LR (image)"),
            ("gain_dir_dssim", "Gain_Dir (1−SSIM)")]
    if "gain_dir_cd" in gains[signals[0]][0]:
        keys.append(("gain_dir_cd", "Gain_Dir (local CD)"))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2))
    for ax, (k, title) in zip(axes, keys):
        for i, s in enumerate(signals):
            vals = [g[k] for g in gains[s]]
            ax.bar(i, np.nanmean(vals), color="#9db4d6", width=0.6)
            ax.scatter([i] * len(vals), vals, color="#1f3b64", s=14, zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        if k == "snr_in":
            ax.axhline(2.0, color="#c0392b", lw=0.8, ls="--")
        ax.set_xticks(range(len(signals)), signals, rotation=20)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "phase0_summary.png", dpi=150)


if __name__ == "__main__":
    main()
