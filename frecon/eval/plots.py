"""D1 figures.                                                  Owner: Tuấn Khải

Required (into <out_dir>/figures/):
  roc.png                ROC curves, one line per signal, random diagonal
  pr_curve.png           precision–recall, prevalence as the random baseline
  failure_enrichment.png enrichment@K for K in {1, 5, 10}%, random = 1.0
  score_vs_error.png     score vs geometric error d_i (hexbin/2D histogram; N is large)

Style: follow the colors and conventions of scripts/make_phase0_figure.py so figures across
the project look the same (same color per signal everywhere).

TODO(TuanKhai): implement each function; one figure per function.
"""
from __future__ import annotations

SIGNAL_COLORS = {  # keep identical across all project figures
    "opacity": "#2a78d6", "gradmag": "#eb6834", "visibility": "#1baf7a",
    "gradient_inconsistency": "#eda100", "random": "#8a8984",
}


def plot_roc(runs, signals, out_path):
    raise NotImplementedError("TODO(TuanKhai)")


def plot_pr(runs, signals, out_path):
    raise NotImplementedError("TODO(TuanKhai)")


def plot_enrichment(metrics_df, out_path):
    raise NotImplementedError("TODO(TuanKhai)")


def plot_score_vs_error(runs, signals, out_path):
    raise NotImplementedError("TODO(TuanKhai)")
