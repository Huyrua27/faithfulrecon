"""D1 evaluation entry point.                                     Owner: Tuấn Khải

Target command (one command, no manual edits):
  python scripts/run_d1.py --scene synthetic --condition v8_az360 [--runs_root runs/d1] \
      [--out_dir results/d1/synthetic/v8_az360] [--n_random 100]

Inputs: every runs/d1/<scene>/<condition>/seed_*/ with scores.npz + geometric_labels.npz.
Outputs:
  <out_dir>/metrics.csv          columns: docs/data_format.md §5
  <out_dir>/figures/             roc.png, pr_curve.png, failure_enrichment.png, score_vs_error.png
  <out_dir>/logs/run_d1.log      inputs (paths + sha256), warnings from validate(), timing

Implemented behavior:
  - discover runs -> load_run -> validate (stop on errors, log warnings)
  - for each seed, signal, label in {f_i, m_i if available}: all_metrics + random_reference
  - write metrics.csv; aggregate selected metrics (mean/std, valid seed count) to the log
  - f_i figures at figures/; available m_i figures at figures/m_i/
  - exit code != 0 if any validation error
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Support the documented direct invocation: python scripts/run_d1.py ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.eval.plots import (
    plot_roc,
    plot_pr,
    plot_enrichment,
    plot_score_vs_error,
)

from frecon.eval.d1_metrics import all_metrics
from frecon.eval.io import discover_runs, load_run, validate
from frecon.eval.random_baseline import random_reference

CSV_FIELDS = [
    "scene",
    "condition",
    "seed",
    "signal",
    "label",
    "prevalence",
    "auroc",
    "auprc",
    "spearman",
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "enrichment_at_1",
    "enrichment_at_5",
    "enrichment_at_10",
    "random_auroc_mean",
    "random_auprc_mean",
    "random_precision_at_5_mean",
]

SUMMARY_METRICS = ("auroc", "auprc", "precision_at_5")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs_root", default="runs/d1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_random", type=int, default=100)
    args = ap.parse_args(argv)
    if args.n_random <= 0:
        ap.error("--n_random must be positive")
    return args


def write_metrics_csv(metric_rows: list[dict], csv_path: Path) -> None:
    if not metric_rows:
        raise ValueError("No metric rows generated")
    for row in metric_rows:
        if set(row) != set(CSV_FIELDS):
            raise ValueError("Metric row does not match CSV schema")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)

        writer.writeheader()

        writer.writerows(metric_rows)


def log_aggregates(metric_rows: list[dict]) -> None:
    groups = defaultdict(list)

    for row in metric_rows:
        key = (row["signal"], row["label"])
        groups[key].append(row)

    for (signal, label), rows in groups.items():
        for metric in SUMMARY_METRICS:
            values = np.array([row[metric] for row in rows], dtype=float)
            valid_values = values[np.isfinite(values)]
            n_valid = len(valid_values)

            if n_valid == 0:
                logging.warning(
                    "Aggregate undefined: signal=%s label=%s metric=%s n_valid=0/%d",
                    signal, label, metric, len(values))
                continue

            valid_seeds = [row["seed"] for row in rows if np.isfinite(row[metric])]
            if n_valid < len(values):
                excluded_seeds = [
                    row["seed"] for row in rows if not np.isfinite(row[metric])
                ]
                logging.warning(
                    "Aggregate excludes undefined seeds: signal=%s label=%s metric=%s "
                    "excluded_seeds=%s",
                    signal,
                    label,
                    metric,
                    excluded_seeds,
                )

            mean = np.mean(valid_values)
            std = np.std(valid_values)

            logging.info(
                "Aggregate: signal=%s label=%s metric=%s mean=%.4f std=%.4f "
                "n_valid=%d/%d seeds=%s",
                signal, label, metric, mean, std, n_valid, len(values), valid_seeds,
            )


def main() -> list[dict]:
    args = parse_args()
    start_time = time.perf_counter()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path("results/d1") / args.scene / args.condition
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "run_d1.log"

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        force=True,
    )

    logging.info(
        "D1 started: scene=%s, condition=%s, n_random=%d",
        args.scene,
        args.condition,
        args.n_random,
    )

    run_dirs = discover_runs(args.runs_root, args.scene, args.condition)
    if not run_dirs:
        logging.error(
            "No valid runs found: root=%s, scene=%s, condition=%s",
            args.runs_root,
            args.scene,
            args.condition,
        )
        raise ValueError("No valid runs found")

    runs = []
    seen_runs = set()

    for run_dir in run_dirs:
        logging.info("Processing run: %s", run_dir)

        try:
            for filename in ("scores.npz", "geometric_labels.npz"):
                input_path = Path(run_dir / filename)
                sha_info = sha256_file(input_path)
                logging.info("Input: path=%s sha256=%s", input_path, sha_info)
            run = load_run(run_dir)
            warnings = validate(run)
            identity = (run.scene, run.condition, run.seed)
            if identity in seen_runs:
                raise ValueError(f"Duplicate run identity: {identity} at {run_dir}")
            seen_runs.add(identity)
        except Exception:
            logging.exception("Failed to process run: %s", run_dir)
            raise
        for warning in warnings:
            logging.warning("Validation warning: %s", warning)

        runs.append(run)

    metric_rows = []

    for run in runs:
        baseline_cache = {}

        for signal_name, score in run.scores.items():
            for label_name in ("f_i", "m_i"):
                if label_name == "m_i" and not run.meta["labels"]["m_i_available"]:
                    continue

                label = run.labels[label_name]

                if label_name not in baseline_cache:
                    random_baseline = random_reference(
                        label=label,
                        error=run.labels["distance"],
                        R=args.n_random,
                        seed=run.seed,
                    )
                    baseline_cache[label_name] = random_baseline
                    for name, (mean, std, q05, q95) in random_baseline.items():
                        logging.info(
                            "Matched random: seed=%s label=%s metric=%s R=%d "
                            "mean=%.8g std=%.8g q05=%.8g q95=%.8g",
                            run.seed, label_name, name, args.n_random, mean, std, q05, q95)

                baseline = baseline_cache[label_name]
                result = all_metrics(score, label, error=run.labels["distance"])

                metadata = {
                    "scene": run.scene,
                    "condition": run.condition,
                    "seed": run.seed,
                    "signal": signal_name,
                    "label": label_name,
                }
                random_fields = {
                    f"random_{metric}_mean": baseline[metric][0]
                    for metric in SUMMARY_METRICS
                }
                combined = {**metadata, **result, **random_fields}
                metric_rows.append(combined)

    signals = sorted({signal_name for run in runs for signal_name in run.scores})

    csv_path = out_dir / "metrics.csv"
    write_metrics_csv(metric_rows, csv_path)

    fig_dir = out_dir / "figures"
    plot_roc(runs, signals, fig_dir / "roc.png")
    plot_pr(runs, signals, fig_dir / "pr_curve.png")
    plot_enrichment(metric_rows, fig_dir / "failure_enrichment.png")
    plot_score_vs_error(runs, signals, fig_dir / "score_vs_error.png")
    if any(run.meta["labels"]["m_i_available"] for run in runs):
        mi_dir = fig_dir / "m_i"
        plot_roc(runs, signals, mi_dir / "roc.png", label_name="m_i")
        plot_pr(runs, signals, mi_dir / "pr_curve.png", label_name="m_i")
        plot_enrichment(
            metric_rows, mi_dir / "failure_enrichment.png", label_name="m_i"
        )
    logging.info(
        "Figures written: %s (root labels=f_i; m_i subdirectory if available)", fig_dir
    )

    log_aggregates(metric_rows)

    elapsed = time.perf_counter() - start_time

    logging.info(
        "D1 completed: rows=%d, elapsed=%.3f seconds", len(metric_rows), elapsed
    )
    return metric_rows


if __name__ == "__main__":
    main()
