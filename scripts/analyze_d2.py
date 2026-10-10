"""Analyze D2 v2 (Phase 0.5) runs: gains vs matched random per step, expected vs opposite
direction, decision rule v2, report.md + figure.

Decision rule v2 (stated before looking at the Phase 0.5 results):
  measurable  : image SNR_in (weight mask) >= snr_min in every seed at the primary step
  per gain    : 'expected' = same sign in every seed, |mean| >= 2*SE, sign as the signal implies (+)
                'opposite' = same, but negative ; otherwise 'inconsistent'
  gains used  : locality  Gain_LR (image, weight mask), Gain_LR (3D displacement)
                direction Gain_Dir (held-out 1-SSIM), Gain_Dir (local accuracy), Gain_Dir (local completeness)
  GO for D2   : at least one measurable signal with an 'expected' or 'opposite' gain
  per signal  : supported      = expected locality AND >= 1 expected direction AND no opposite direction
                not supported  = >= 1 opposite direction
                no evidence    = otherwise (or not measurable)

Usage: python scripts/analyze_d2.py --run_dir runs/phase0_5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

GAINS = [  # key, label, group
    ("lr_img", "Gain_LR image (weight mask)", "locality"),
    ("lr_param", "Gain_LR 3D displacement", "locality"),
    ("dir_dssim", "Gain_Dir held-out 1−SSIM", "direction"),
    ("dir_acc", "Gain_Dir local accuracy", "direction"),
    ("dir_comp", "Gain_Dir local completeness", "direction"),
]


def _nanmean(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def raw(sel: dict, t: str) -> dict:
    s = sel["by_step"][t]
    out = {"lr_img": s["image"]["weight"]["LR"], "lr_img_fp": s["image"]["footprint"]["LR"],
           "lr_param": s["param"]["LR"], "dir_dssim": -s["heldout"]["dssim"],
           "snr": s["image"]["weight"].get("snr_in", float("nan")),
           "snr_fp": s["image"]["footprint"].get("snr_in", float("nan"))}
    if "geometry" in s:
        out["dir_acc"] = -s["geometry"]["d_acc"]
        out["dir_comp"] = -s["geometry"]["d_comp"]
    return out


def seed_gains(m: dict, signal: str, t: str) -> dict:
    rnd = [raw(v, t) for v in m["selections"].values() if v["kind"] == "random"]
    s = raw(m["selections"][signal], t)
    g = {k: s[k] - _nanmean([r[k] for r in rnd]) for k, _, _ in GAINS if k in s}
    g.update(snr=s["snr"], snr_random=_nanmean([r["snr"] for r in rnd]), lr_img_abs=s["lr_img"],
             lr_img_random=_nanmean([r["lr_img"] for r in rnd]), lr_fp_abs=s["lr_img_fp"],
             lr_fp_random=_nanmean([r["lr_img_fp"] for r in rnd]))
    return g


def classify(vals) -> str:
    v = np.asarray([x for x in vals if not math.isnan(x)])
    if v.size < 2:
        return "n/a"
    if (np.all(v > 0) or np.all(v < 0)) and abs(v.mean()) >= 2 * v.std(ddof=1) / math.sqrt(v.size):
        return "expected" if v.mean() > 0 else "opposite"
    return "inconsistent"


def fmt(vals, scale=1.0, d=3):
    v = np.asarray([x for x in vals if not math.isnan(x)]) * scale
    if v.size == 0:
        return "n/a"
    return f"{v.mean():+.{d}f} ± {v.std(ddof=1):.{d}f}" if v.size > 1 else f"{v[0]:+.{d}f}"


def analyze(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ms = [json.loads(p.read_text()) for p in sorted(run_dir.glob("seed_*/metrics.json"))]
    if not ms:
        raise SystemExit(f"no metrics under {run_dir}")
    steps = [str(t) for t in ms[0]["steps"]]
    prim = str(cfg.get("analysis", {}).get("primary_step", steps[-1]))
    if prim not in steps:
        prim = steps[-1]
    snr_min = cfg.get("analysis", {}).get("snr_min", 2.0)
    signals = [k for k, v in ms[0]["selections"].items() if v["kind"] == "signal"]
    G = {s: {t: [seed_gains(m, s, t) for m in ms] for t in steps} for s in signals}
    rnd_snr = {t: [_nanmean([raw(v, t)["snr"] for v in m["selections"].values() if v["kind"] == "random"])
                   for m in ms] for t in steps}
    meas = {s: all(g["snr"] >= snr_min for g in G[s][prim]) for s in signals}
    cls = {s: {t: {k: classify([g[k] for g in G[s][t]]) for k, _, _ in GAINS if k in G[s][t][0]}
               for t in steps} for s in signals}
    return cfg, ms, steps, prim, snr_min, signals, G, rnd_snr, meas, cls


def verdicts(signals, prim, meas, cls):
    out, go = {}, False
    for s in signals:
        c = cls[s][prim]
        loc = [c.get(k) for k, _, grp in GAINS if grp == "locality"]
        dirs = [c.get(k) for k, _, grp in GAINS if grp == "direction"]
        if meas[s] and any(x in ("expected", "opposite") for x in c.values()):
            go = True
        if not meas[s]:
            out[s] = "not measurable"
        elif "opposite" in dirs:
            out[s] = "stated meaning not supported (direction opposite to expected)"
        elif "expected" in loc and "expected" in dirs:
            out[s] = "supported (localized and in the expected direction)"
        else:
            out[s] = "no evidence either way"
    return go, out


def report(run_dir: Path):
    cfg, ms, steps, prim, snr_min, signals, G, rnd_snr, meas, cls = analyze(run_dir)
    go, verdict = verdicts(signals, prim, meas, cls)
    ic = cfg["intervention"]
    L = ["# Phase 0.5 — D2 protocol v2", "",
         f"- Seeds: {[m['seed'] for m in ms]} · K = {ic['frac']:.0%} · removal · snapshots {steps} steps · "
         f"{ic['n_random']} random selections per seed · checkpoints and selections identical to Phase 0",
         f"- Weight mask: selected Gaussians carry ≥ {ic['region_weight_thresh']:.0%} of the pixel's blending weight "
         f"(footprint mask of Phase 0 reported for comparison)",
         "- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.",
         "- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent", ""]

    L += ["## Region size and image locality", "", "| signal | region (weight) | region (footprint) | "
          f"LR_img weight @ {prim} (random) | LR_img footprint @ {prim} (random) |", "|---|---|---|---|---|"]
    for s in signals:
        rf = [m["selections"][s]["region_frac"] for m in ms]
        rr = [_nanmean([v["region_frac"]["weight"] for v in m["selections"].values() if v["kind"] == "random"]) for m in ms]
        g = G[s][prim]
        L.append(f"| {s} | {np.mean([r['weight'] for r in rf]):.2%} (random {np.mean(rr):.2%}) | "
                 f"{np.mean([r['footprint'] for r in rf]):.2%} | {_nanmean([x['lr_img_abs'] for x in g]):.3f} "
                 f"({_nanmean([x['lr_img_random'] for x in g]):.3f}) | {_nanmean([x['lr_fp_abs'] for x in g]):.3f} "
                 f"({_nanmean([x['lr_fp_random'] for x in g]):.3f}) |")

    L += ["", "## Measurability (image SNR_in, weight mask)", "",
          "| signal | " + " | ".join(f"step {t}" for t in steps if t != "0") + " | measurable |",
          "|---|" + "---|" * (len(steps) - (1 if "0" in steps else 0)) + "---|"]
    for s in signals:
        L.append(f"| {s} | " + " | ".join(fmt([g['snr'] for g in G[s][t]], d=1) for t in steps if t != "0")
                 + f" | {'yes' if meas[s] else 'no'} |")
    L.append("| random | " + " | ".join(fmt(rnd_snr[t], d=1) for t in steps if t != "0") + " | |")

    for t in steps:
        L += ["", f"## Gains at step {t}" + (" (right after removal, no re-optimization)" if t == "0" else ""), "",
              "| signal | " + " | ".join(lbl for _, lbl, _ in GAINS) + " |", "|---|" + "---|" * len(GAINS)]
        for s in signals:
            cells = []
            for k, _, _ in GAINS:
                if k not in G[s][t][0]:
                    cells.append("n/a")
                    continue
                scale = 1e3 if k in ("dir_dssim", "dir_acc", "dir_comp") else 1.0
                c = cls[s][t].get(k, "n/a")
                tag = {"expected": " **E**", "opposite": " **O**"}.get(c, "")
                cells.append(fmt([g[k] for g in G[s][t]], scale) + tag)
            L.append(f"| {s}{'' if meas[s] else ' (n.m.)'} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Direction gains ×10⁻³.")

    L += ["", "## Decision", "", f"**D2: {'GO' if go else 'NO-GO'}** (rule at the top of scripts/analyze_d2.py, primary step {prim}).", ""]
    for s in signals:
        L.append(f"- **{s}**: {verdict[s]}")
    (run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    figure(run_dir, cfg, steps, prim, signals, G, rnd_snr, meas, cls, go, verdict)


def figure(run_dir, cfg, steps, prim, signals, G, rnd_snr, meas, cls, go, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from make_phase0_figure import COLORS, INK, INK2, RANDOM, SURFACE, THRESH, style  # noqa: F401

    x = list(range(len(steps)))  # categorical: steps are unevenly spaced
    panels = [("snr", "A  Response vs noise floor\n    SNR_in (weight mask)", 1.0)] + \
             [(k, f"{chr(66 + i)}  {lbl}", 1e3 if g == "direction" else 1.0) for i, (k, lbl, g) in enumerate(GAINS)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.21, hspace=0.62, wspace=0.28)
    ic = cfg["intervention"]
    fig.text(0.05, 0.965, "FaithfulRecon — Phase 0.5: D2 protocol v2", fontsize=15, weight="bold", color=INK)
    fig.text(0.05, 0.935, f"Same checkpoints and selections as Phase 0 · remove top-{ic['frac']:.0%} · response right after "
             f"removal and after {', '.join(steps[1:])} re-optimization steps · {len(G[signals[0]][steps[0]])} seeds\n"
             "Gain = signal − mean of size-matched random removals (same seed); > 0 = more local / better than random. "
             "Dashed = not measurable. E / O = consistent expected / opposite sign at that step.",
             fontsize=9, color=INK2, va="top", linespacing=1.6)
    for ax, (k, title, scale) in zip(axes.flat, panels):
        for i, s in enumerate(signals):
            col = COLORS.get(s, "#4a3aa7")
            vals = np.array([[g.get(k, np.nan) * scale for g in G[s][t]] for t in steps])
            if k == "snr":
                vals[[j for j, t in enumerate(steps) if t == "0"]] = np.nan
            m, sd = np.nanmean(vals, 1), np.nanstd(vals, 1, ddof=1)
            ls = "-" if meas[s] else "--"
            off = (i - 1) * 0.06
            xx = np.array(x) + off
            ax.errorbar(xx, m, yerr=sd, color=col, ls=ls, lw=2, marker="o", ms=5, capsize=3,
                        label=s, alpha=1 if meas[s] else 0.5, mec=SURFACE, mew=1)
            if k != "snr":
                for j, t in enumerate(steps):
                    c = cls[s][t].get(k)
                    if c in ("expected", "opposite") and meas[s] and not np.isnan(m[j]):
                        ax.annotate("E" if c == "expected" else "O", (xx[j], m[j]), xytext=(4, 4),
                                    textcoords="offset points", fontsize=8, weight="bold",
                                    color="#1b7f3b" if c == "expected" else THRESH)
        if k == "snr":
            r = np.array(rnd_snr[steps[-1]])
            rv = np.array([[v for v in rnd_snr[t]] for t in steps])
            rv[[j for j, t in enumerate(steps) if t == "0"]] = np.nan
            ax.errorbar(x, np.nanmean(rv, 1), yerr=np.nanstd(rv, 1, ddof=1), color=RANDOM, lw=2, marker="o",
                        ms=5, capsize=3, label="random")
            ax.axhline(cfg.get("analysis", {}).get("snr_min", 2.0), color=THRESH, lw=1, ls="--")
            ax.set_yscale("log")
            style(ax, zero=False)
        else:
            style(ax)
        ax.set_title(title + ("\n    (×10⁻³)" if scale != 1.0 else "\n "), linespacing=1.5)
        ax.set_xticks(x, steps)
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.set_xlabel("re-optimization steps after removal")
    axes.flat[0].legend(frameon=False, fontsize=8, loc="best")
    lines = [f"D2 decision (primary step {prim}): {'GO' if go else 'NO-GO'}"] + \
            [f"• {s}: {verdict[s]}" for s in signals]
    fig.text(0.05, 0.03, "\n".join(lines), fontsize=9.5, color=INK, va="bottom", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f1f0ec", edgecolor="none"))
    out = run_dir / "phase0_5_figure.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/phase0_5")
    report(Path(ap.parse_args().run_dir))


if __name__ == "__main__":
    main()
