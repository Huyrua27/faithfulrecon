"""Compute geometric labels for a checkpoint.                     Owner: Trường Thịnh

Target command:
  python scripts/make_labels.py --run_dir runs/d1/synthetic/v8_az360/seed_0 \
      --config configs/phase0.yaml --tau 0.03 --k_views 2 --theta_deg 10

Outputs (docs/data_format.md §4):
  <run_dir>/geometric_labels.npz          (replaces the provisional file; provisional=False)
  results/benchmark/label_stats.csv       prevalence of m_i / f_i per threshold setting
  results/benchmark/figures/*.png         figures listed in frecon/benchmark/visualize_labels.py

TODO(TruongThinh):
  - [ ] load checkpoint (checkpoint.json) + scene cameras + reference geometry
  - [ ] reference_geometry.check_alignment() must pass first (print the result)
  - [ ] distance = geometric_error.point_to_surface_distance(...)
  - [ ] m_i = failure_labels.condition_labels(...); f_i = failure_labels.outcome_labels(...)
  - [ ] failure_labels.save_labels(...) with m_i_available=True, provisional=False
  - [ ] --sweep: loop over several (tau, k_views, theta_deg) and write label_stats.csv
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--k_views", type=int, default=2)
    ap.add_argument("--theta_deg", type=float, default=10.0)
    ap.add_argument("--sweep", action="store_true")
    ap.parse_args()
    raise NotImplementedError("TODO(TruongThinh): see module docstring")


if __name__ == "__main__":
    main()
