"""D3 evaluation entry point.                                     Owner: Phát

Target command:
  python scripts/run_d3.py --scene synthetic --condition v8_az360 --config configs/phase0.yaml \
      [--runs_root runs/d1] [--out_dir results/d3/synthetic/v8_az360]

Inputs: runs/d1/<scene>/<condition>/seed_*/ (checkpoint.json, scores.npz, geometric_labels.npz)
        + held-out cameras from the scene config.
Outputs: <out_dir>/d3_metrics.csv, <out_dir>/figures/ (spec: docs/tasks/Phat_tasks.md §2.3)

TODO(Phat): load -> e_i, score maps -> metrics per (seed, signal) + random -> csv + figures.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--runs_root", default="runs/d1")
    ap.add_argument("--out_dir", default=None)
    ap.parse_args()
    raise NotImplementedError("TODO(Phat): see module docstring")


if __name__ == "__main__":
    main()
