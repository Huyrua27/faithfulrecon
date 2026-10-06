"""D1 evaluation entry point.                                     Owner: Tuấn Khải

Target command (one command, no manual edits):
  python scripts/run_d1.py --scene synthetic --condition v8_az360 [--runs_root runs/d1] \
      [--out_dir results/d1/synthetic/v8_az360] [--n_random 100]

Inputs: every runs/d1/<scene>/<condition>/seed_*/ with scores.npz + geometric_labels.npz.
Outputs:
  <out_dir>/metrics.csv          columns: docs/data_format.md §5
  <out_dir>/figures/             roc.png, pr_curve.png, failure_enrichment.png, score_vs_error.png
  <out_dir>/logs/run_d1.log      inputs (paths + sha256), warnings from validate(), timing

TODO(TuanKhai):
  - [ ] discover runs -> load_run -> validate (stop on errors, log warnings)
  - [ ] for each seed, signal, label in {f_i, m_i if available}: all_metrics + random_reference
  - [ ] write metrics.csv; aggregate table (mean ± std over seeds) printed to the log
  - [ ] figures via frecon/eval/plots.py
  - [ ] exit code != 0 if any validation error
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--runs_root", default="runs/d1")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_random", type=int, default=100)
    ap.parse_args()
    raise NotImplementedError("TODO(TuanKhai): see module docstring")


if __name__ == "__main__":
    main()
