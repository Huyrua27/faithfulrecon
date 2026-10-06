"""Compute reliability signals on a checkpoint and export them.      Owner: Quỳnh Anh

Target command:
  python scripts/export_signals.py --run_dir runs/d1/synthetic/v8_az360/seed_0 \
      --config configs/phase0.yaml --signals opacity gradmag visibility gradient_inconsistency random

Outputs (docs/data_format.md §3):
  <run_dir>/scores.npz
  <run_dir>/signal_results.csv           (long format; stays in runs/, not committed)
  results/signals/figures/<scene>_<condition>_seed<s>_<signal>_top{1,5,10}.png

TODO(QuynhAnh):
  - [ ] read checkpoint.json, load the model (Trainer.load), load training cams (load_scene)
  - [ ] frecon.signals.compute_signals(...) with the requested names (one shared context)
  - [ ] write scores.npz with meta (scene, condition, seed, checkpoint_sha256, n_gaussians, signal_params)
  - [ ] write signal_results.csv (scene,condition,seed,signal,gaussian_id,score)
  - [ ] Top-K visualization: render a training/held-out view with the top-K Gaussians colored
        (model.render(..., features=<color>)), or a 3D scatter of their centers, K = 1/5/10 %
  - [ ] print a short summary per signal (min/median/max, #ties, #zero-visibility Gaussians)
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--signals", nargs="+", default=["random", "opacity", "gradmag", "visibility"])
    ap.add_argument("--no_csv", action="store_true")
    ap.parse_args()
    raise NotImplementedError("TODO(QuynhAnh): see module docstring")


if __name__ == "__main__":
    main()
