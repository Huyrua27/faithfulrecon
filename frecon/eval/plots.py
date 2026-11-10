"""D1 batch figures for validated paired runs.

Curves average equally over defined seeds for each signal and label. Precision-recall
uses empirical steps, matching the average-precision metric. Root figures default to
f_i; callers can request m_i, which skips runs where that label is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from frecon.eval.d1_metrics import K_FRACTIONS

SIGNAL_COLORS = {
    "opacity": "#2a78d6",
    "gradmag": "#eb6834",
    "visibility": "#1baf7a",
    "gradient_inconsistency": "#eda100",
    "random": "#8a8984",
}


def _defined_runs(runs, signal, label_name):
    for run in runs:
        if signal not in run.scores:
            continue
        if label_name == "m_i" and not run.meta["labels"]["m_i_available"]:
            continue
        label = run.labels[label_name]
        if np.all(label) or not np.any(label):
            logging.warning(
                "Plot skips degenerate label: seed=%s signal=%s label=%s",
                run.seed,
                signal,
                label_name,
            )
            continue
        yield run


def _save(fig, out_path):
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def _no_data(ax, description):
    ax.text(
        0.5,
        0.5,
        f"No defined {description}",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def plot_roc(runs, signals, out_path, *, label_name="f_i"):
    runs = tuple(runs)
    fig, ax = plt.subplots(figsize=(7, 5))
    n_curves = 0

    for signal in signals:
        curves = [
            roc_curve(run.labels[label_name], run.scores[signal])[:2]
            for run in _defined_runs(runs, signal, label_name)
        ]

        if not curves:
            continue
        grid = np.unique(np.concatenate([fpr for fpr, _ in curves]))
        lower_tprs, upper_tprs = [], []
        for fpr, tpr in curves:
            # Preserve both ends of every vertical segment. Dropping one end
            # can turn a perfect or inverted ROC into a misleading diagonal.
            unique_fpr, indices, counts = np.unique(
                fpr,
                return_index=True,
                return_counts=True,
            )
            upper = np.interp(grid, fpr, tpr)
            lower = upper.copy()
            grid_indices = np.searchsorted(grid, unique_fpr)
            lower[grid_indices] = tpr[indices]
            upper[grid_indices] = tpr[indices + counts - 1]
            lower_tprs.append(lower)
            upper_tprs.append(upper)
        ax.plot(
            np.repeat(grid, 2),
            np.column_stack(
                (
                    np.mean(lower_tprs, axis=0),
                    np.mean(upper_tprs, axis=0),
                )
            ).ravel(),
            color=SIGNAL_COLORS.get(signal, "#333333"),
            linewidth=2,
            label=f"{signal} (n={len(curves)})",
        )
        n_curves += 1

    ax.plot([0, 1], [0, 1], "--", color=SIGNAL_COLORS["random"], label="Random")
    if not n_curves:
        _no_data(ax, f"{label_name} ROC curves")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC — {label_name} (mean across seeds)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    _save(fig, out_path)


def plot_pr(runs, signals, out_path, *, label_name="f_i"):
    runs = tuple(runs)
    fig, ax = plt.subplots(figsize=(7, 5))
    baselines = []

    for signal in signals:
        curves = []
        prevalences = []
        for run in _defined_runs(runs, signal, label_name):
            label = run.labels[label_name]
            precision, recall, _ = precision_recall_curve(label, run.scores[signal])
            # For each recall level, retain precision at the first threshold
            # reaching it. Negative-only threshold changes have zero recall width.
            unique_recall, indices = np.unique(recall[::-1], return_index=True)
            curves.append((unique_recall, precision[::-1][indices]))
            prevalences.append(float(label.mean()))

        if not curves:
            continue

        # The union preserves every seed's recall step, so the area under the
        # mean step curve equals mean AP (rather than a linear interpolation).
        recall_grid = np.unique(np.concatenate([recall for recall, _ in curves]))
        all_precisions = [
            precision[np.searchsorted(recall, recall_grid, side="left")]
            for recall, precision in curves
        ]
        mean_prevalence = float(np.mean(prevalences))
        ax.step(
            np.r_[recall_grid, 1.0],
            np.r_[np.mean(all_precisions, axis=0), mean_prevalence],
            where="pre",
            color=SIGNAL_COLORS.get(signal, "#333333"),
            linewidth=2,
            label=f"{signal} (n={len(curves)})",
        )
        baselines.append((signal, mean_prevalence))

    if baselines:
        if all(np.isclose(value, baselines[0][1]) for _, value in baselines):
            ax.axhline(
                baselines[0][1],
                linestyle="--",
                color=SIGNAL_COLORS["random"],
                label=f"Random (mean prevalence={baselines[0][1]:.3g})",
            )
        else:
            for signal, prevalence in baselines:
                ax.axhline(
                    prevalence,
                    linestyle="--",
                    color=SIGNAL_COLORS.get(signal, "#333333"),
                    label=f"Random for {signal} ({prevalence:.3g})",
                )
    else:
        _no_data(ax, f"{label_name} precision-recall curves")

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision–Recall — {label_name} (mean across seeds)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if ax.lines:
        ax.legend()
    _save(fig, out_path)


def plot_enrichment(metrics_df, out_path, *, label_name="f_i"):
    """Accept metric row records or a DataFrame without requiring pandas."""
    records = (
        metrics_df.to_dict(orient="records")
        if hasattr(metrics_df, "to_dict")
        else list(metrics_df)
    )
    records = [row for row in records if row["label"] == label_name]
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [round(fraction * 100) for fraction in K_FRACTIONS]
    n_curves = 0

    for signal in sorted({row["signal"] for row in records}):
        values = np.array(
            [
                [row[f"enrichment_at_{k}"] for k in ks]
                for row in records
                if row["signal"] == signal
            ],
            dtype=float,
        )
        valid = np.isfinite(values)
        counts = valid.sum(axis=0)
        if not counts.any():
            continue
        means = np.divide(
            np.where(valid, values, 0).sum(axis=0),
            counts,
            out=np.full(len(ks), np.nan),
            where=counts > 0,
        )
        n_label = (
            str(counts[0])
            if np.all(counts == counts[0])
            else f"{counts.min()}–{counts.max()}"
        )
        ax.plot(
            ks,
            means,
            marker="o",
            linewidth=2,
            color=SIGNAL_COLORS.get(signal, "#333333"),
            label=f"{signal} (n={n_label})",
        )
        n_curves += 1

    ax.axhline(1.0, linestyle="--", color=SIGNAL_COLORS["random"], label="Random")
    if not n_curves:
        _no_data(ax, f"{label_name} enrichment values")
    ax.set(
        xlabel="Top K (%)",
        ylabel="Enrichment@K",
        title=f"{'Failure' if label_name == 'f_i' else 'Condition'} enrichment — {label_name} (mean across seeds)",
        xticks=ks,
        xlim=(min(ks) - 0.5, max(ks) + 0.5),
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    _save(fig, out_path)


def plot_score_vs_error(runs, signals, out_path):
    runs = tuple(runs)
    signals = tuple(signals)
    if not signals:
        raise ValueError("No signals to plot")
    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=(7, 4 * len(signals)),
        squeeze=False,
    )
    for ax, signal in zip(axes[:, 0], signals):
        selected = [run for run in runs if signal in run.scores]
        if not selected:
            _no_data(ax, f"{signal} score/error data")
            continue
        errors = np.concatenate([run.labels["distance"] for run in selected])
        scores = np.concatenate([run.scores[signal] for run in selected])
        hb = ax.hexbin(
            errors,
            scores,
            gridsize=60,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        fig.colorbar(hb, ax=ax, label="Gaussian count (log color scale)")
        ax.set(
            xlabel="Geometric error d_i",
            ylabel="Risk score (higher = less reliable)",
            title=f"{signal} (pooled over {len(selected)} seeds)",
        )

    fig.tight_layout()
    _save(fig, out_path)
