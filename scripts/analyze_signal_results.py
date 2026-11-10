"""Quỳnh Anh's signal audit and Phase 0 interpretation; not the production D1 evaluator.

Reads existing metrics/labels without changing the D2 protocol or provisional labels.
Uses sklearn/scipy reference metrics for exploratory signal sanity checks only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from analyze_phase0 import per_seed_gains
from export_signals import check_identity, read_archive, selected_ids


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values, digits=4):
    values = np.array(list(values), dtype=float)
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--scores_root", type=Path, default=Path("runs/signals/recomputed/synthetic/v8_az360"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_dir", type=Path, default=Path("results/signals"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparisons, sanity, d2, counts, confounders = [], [], [], [], []
    for seed in args.seeds:
        supplied = args.data_root / "d1/synthetic/v8_az360" / f"seed_{seed}"
        computed = args.scores_root / f"seed_{seed}"
        original, original_meta = read_archive(supplied / "scores.npz")
        scores, meta = read_archive(computed / "scores.npz")
        labels, label_meta = read_archive(supplied / "geometric_labels.npz")
        diagnostics, diagnostic_meta = read_archive(computed / "diagnostics.npz")
        digest, n = meta["checkpoint_sha256"], meta["n_gaussians"]
        for description, identity in (("original scores", original_meta), ("labels", label_meta),
                                      ("diagnostics", diagnostic_meta)):
            check_identity(identity, digest, n, description)
        if label_meta.get("m_i_available", False):
            raise ValueError("this pilot audit expects provisional f_i only; use the D1 evaluator for final labels")
        counts.append(dict(seed=seed, n_gaussians=n, n_zero_visibility=int((diagnostics["vis_count"] == 0).sum()),
                           n_gi_insufficient_views=int((diagnostics["gi_count"] < 2).sum()),
                           checkpoint_sha256=digest))
        gi = scores["gradient_inconsistency"]
        gi_ids = selected_ids(gi, .05)
        effective_delta = meta["signal_params"]["gradient_inconsistency"]["effective_delta"]
        confounders.append(dict(seed=seed,
            spearman_gi_opacity_score=float(spearmanr(gi, scores["opacity"]).statistic),
            spearman_gi_visibility_count=float(spearmanr(gi, diagnostics["vis_count"]).statistic),
            spearman_gi_gradmag=float(spearmanr(gi, scores["gradmag"]).statistic),
            median_mean_gradient_sq=float(np.median(diagnostics["gi_mean_norm_sq"][diagnostics["gi_count"] >= 2])),
            top5_median_mean_gradient_sq=float(np.median(diagnostics["gi_mean_norm_sq"][gi_ids])),
            top5_fraction_mean_sq_below_delta=float((diagnostics["gi_mean_norm_sq"][gi_ids] < effective_delta).mean())))
        for name, score in scores.items():
            if name in original:
                previous = original[name]
                ids, old_ids = selected_ids(score, .05), selected_ids(previous, .05)
                row = dict(seed=seed, signal=name, max_abs_difference=float(abs(score - previous).max()),
                           spearman=float(spearmanr(score, previous).statistic),
                           overlap_at_5=float(np.intersect1d(ids, old_ids).size / ids.size),
                           visibility_count_mismatches="")
                if name == "visibility":
                    # The documented legacy score is -count + U[0, .001).
                    original_count = np.ceil(-previous).astype(int)
                    row["visibility_count_mismatches"] = int((original_count != diagnostics["vis_count"]).sum())
                comparisons.append(row)
            auc = float(roc_auc_score(labels["f_i"], score))
            for frac in (.01, .05, .10):
                ids = selected_ids(score, frac)
                precision = float(labels["f_i"][ids].mean())
                sanity.append(dict(seed=seed, signal=name, top_percent=round(frac * 100),
                    n_selected=len(ids), auroc_f_i=auc, prevalence=float(labels["f_i"].mean()),
                    precision=precision, enrichment=precision / float(labels["f_i"].mean()),
                    zero_visibility_fraction=float((diagnostics["vis_count"][ids] == 0).mean()),
                    gi_insufficient_fraction=float((diagnostics["gi_count"][ids] < 2).mean()),
                    provisional_labels=bool(label_meta.get("provisional", False))))
        metrics = json.loads((args.data_root / f"seed_{seed}/metrics.json").read_text())
        if metrics["n_gaussians"] != n or metrics["seed"] != seed:
            raise ValueError("Phase 0 metrics identity differs from exported run")
        for name, selection in metrics["selections"].items():
            if selection["kind"] == "signal":
                d2.append(dict(seed=seed, signal=name, **per_seed_gains(metrics, name),
                               region_fraction=selection["image"]["region_frac"]))
    write_csv(args.out_dir / "checkpoint_audit.csv", counts)
    write_csv(args.out_dir / "reproduction_comparison.csv", comparisons)
    write_csv(args.out_dir / "signal_sanity.csv", sanity)
    write_csv(args.out_dir / "phase0_per_seed.csv", d2)
    write_csv(args.out_dir / "confounder_checks.csv", confounders)
    lines = ["# Signal audit — Quỳnh Anh", "", "Date: 2026-10-16 (Asia/Bangkok).",
        "", "Scene: synthetic / v8_az360; supplied checkpoints, seeds " + str(args.seeds) + ".",
        "Recomputed scores use CPU / torch renderer and results/phase0/config_used.yaml.",
        "Checkpoint hashes, Gaussian counts, score types and finiteness are checked by the exporter.",
        "", "## Checkpoint and evidence counts", "",
        "| Seed | N | No training-view contribution | Fewer than 2 contributing views |",
        "|---|---:|---:|---:|"]
    for c in counts:
        lines.append(f"| {c['seed']} | {c['n_gaussians']} | {c['n_zero_visibility']} | {c['n_gi_insufficient_views']} |")
    lines += ["", "## Reproduction of supplied scores", "",
        "Overlap is intersection / K between supplied and recomputed Top 5% in the SAME checkpoint.",
        "Exact score ties in the exporter use Gaussian ID. This does not replay Phase 0 torch.topk ties.",
        "", "| Seed | Signal | Max absolute difference | Spearman | Top 5% overlap | Count mismatches |",
        "|---|---|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"| {c['seed']} | {c['signal']} | {c['max_abs_difference']:.3g} | "
                     f"{c['spearman']:.6f} | {c['overlap_at_5']:.3%} | {c['visibility_count_mismatches']} |")
    lines += ["", "### Interpretation of reproduction differences", "",
        "Opacity is reproduced within float32 activation precision. GradMag rankings and Top 5% are nearly identical, "
        "but some gradients differ appreciably in value: this is a ranking reproduction, not bitwise score reproduction.",
        "Random-score differences are consistent with different device RNG streams (the saved config requests CUDA; "
        "the original runtime/version is not recorded). The source runtime has not been verified; no exact RNG reproduction is claimed.",
        "Visibility differences are predominantly within tied integer counts; the separate count-mismatch column flags "
        "threshold-sensitive numerical differences. Device/renderer arithmetic and threshold sensitivity remain audit items.",
        "", "## Exploratory association with provisional f_i", "",
        "Mean ± sample std across seeds. These are sanity checks with supplied provisional labels, not final D1 results.",
        "Random expected precision equals prevalence; the exported random vector is one draw, not the matched D2 control.",
        "", "| Signal | AUROC | Precision@5% | Enrichment@5% | Zero-visibility fraction in Top 5% |",
        "|---|---:|---:|---:|---:|"]
    names = list(dict.fromkeys(r["signal"] for r in sanity))
    for name in names:
        rows = [r for r in sanity if r["signal"] == name and r["top_percent"] == 5]
        lines.append(f"| {name} | " + " | ".join(mean_std(r[k] for r in rows) for k in
            ("auroc_f_i", "precision", "enrichment", "zero_visibility_fraction")) + " |")
    lines += ["", "Gradient inconsistency shows association with provisional floater labels, but no D2 intervention "
        "for this new signal has been run. Association does not establish removal benefit or faithfulness.",
        "Opacity's zero Top-5% floater precision must be interpreted with the label gate opacity > .05.",
        "The count<2 fallback is zero evidence of inconsistency, not evidence of reliable geometry.",
        "", "## Gradient inconsistency: initial confounder checks", "",
        "| Seed | Spearman vs opacity score (1-alpha) | vs visibility count | vs GradMag | Top 5% mean-gradient squared < delta |",
        "|---|---:|---:|---:|---:|"]
    for r in confounders:
        lines.append(f"| {r['seed']} | {r['spearman_gi_opacity_score']:.3f} | "
            f"{r['spearman_gi_visibility_count']:.3f} | {r['spearman_gi_gradmag']:.3f} | "
            f"{r['top5_fraction_mean_sq_below_delta']:.2%} |")
    lines += ["", "The moderate positive association with visibility count deserves a stratified ablation. "
        "Marginal correlations do not establish domination or independence. Near-zero mean gradients occur "
        "in a small portion of Top 5%; delta sensitivity still requires an explicit sweep.",
        "", "## Phase 0: original interventions, original selections", "",
        "These values are read from supplied metrics.json; interventions were NOT rerun using CPU selections.",
        "Measurable requires image SNR_in >= 2 in every seed. Gains below that gate are not interpreted.",
        "", "| Signal | Seed | SNR_in | Gain LR (3D) | Gain Dir (local CD) | Measurable in this seed |",
        "|---|---:|---:|---:|---:|---|"]
    for r in d2:
        lines.append(f"| {r['signal']} | {r['seed']} | {r['snr_in']:.3f} | {r['gain_LR_param']:+.6f} | "
                     f"{r['gain_dir_cd']:+.8f} | {'yes' if r['snr_in'] >= 2 else 'no'} |")
    lines += ["", "### Observation", "",
        "GradMag is measurable in all seeds; 3D locality gains are positive and local-Chamfer direction gains negative. "
        "Opacity is near the noise floor. Visibility does not pass the measurability gate in all seeds.",
        "", "### Interpretation", "",
        "GradMag selects influential primitives with some floater enrichment, but their removal harms local geometry "
        "relative to matched random removal in this pilot. A possible explanation is mixed selection of erroneous "
        "and necessary structural Gaussians; this explanation needs structural inspection, not only AUROC.",
        "Image LR is high for random removal as well; high LR alone does not show discriminating locality. "
        "Separate 3D locality and mask coverage are required. Three seeds support pilot observations only.",
        "", "### Limitations and next actions", "",
        "- The supplied m_i is unavailable and all False; no m_i evaluation is performed.",
        "- Original execution environment and config provenance are incomplete; the committed saved config is an assumption.",
        "- Do not compare Gaussian IDs across independently trained seeds. Compare metrics/distributions across seeds.",
        "- Keep existing D2 unchanged. Propose immediate-removal measurement and narrower masks to the D2 owner.",
        "- Test visibility tie seeds, contribution threshold, and GI delta in separate ablations.",
        "- Test whether GI ranking is dominated by visibility, opacity or near-zero mean gradient before interpretation.", ""]
    (args.out_dir / "QuynhAnh_signal_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote signal audit and five summary CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
