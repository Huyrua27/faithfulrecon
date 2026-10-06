# Phase 0 — D2 proof of concept

Phase 0 decides whether the intervention protocol (D2) produces a response that can be
separated from re-optimization noise before the full study is run.
Status: **done (GO)** — see `results/phase0/` and `docs/analysis/phase0_analysis.md`.

## Running

```bash
python scripts/phase0.py --config configs/phase0.yaml --quick --seeds 0 1   # smoke test, a few minutes
python scripts/phase0.py --config configs/phase0.yaml                         # full Phase 0 (~2 h on a GTX 1650)
python scripts/analyze_phase0.py --run_dir runs/phase0                        # report.md + go/no-go
python scripts/make_phase0_figure.py --run_dir runs/phase0                    # report figures
```

Every stage is cached under `runs/phase0/seed_<s>/` (`base.pt`, `signals.pt`, `run_*.pt`,
`metrics.json`), so an interrupted run resumes where it stopped. Delete a file to recompute it.
To run seeds on different machines: `--seeds 0`, `--seeds 1`, ... then analyze the merged dir.

## What Phase 0 does

Per seed:

1. Train a base 3DGS model from a random init on the sparse training views.
2. Compute signals on the base model and select the top-K (K = 5%) least reliable
   Gaussians for each signal, plus `n_random` size-matched random selections (the control).
3. For each selection: remove it, re-optimize `reopt_steps` with densification off at the
   final learning rate, render the held-out views.
4. Two null interventions (no removal) with different view orders: `null_a` is the paired
   reference for every selection (same re-optimization seed), `null_b` gives the noise floor.

Measured against `null_a`, for each selection:

| Quantity | Definition |
|---|---|
| `d_in`, `d_out` | mean \|I_int − I_null_a\| inside / outside the projected region of the selection (region = selected Gaussians rendered alone at opacity 0.99, alpha > 0.5) |
| `LR` (image) | d_in / (d_in + d_out) — both are per-pixel means, so this is the size-normalized ratio |
| `snr_in` | d_in / d_in(null_b vs null_a) on the same region: effect relative to the noise floor |
| `LR` (param) | same ratio for the displacement of surviving Gaussian centers within / beyond `neighborhood_radius` of the removed ones |
| `heldout` | change of held-out L1, 1−SSIM, PSNR vs null_a (D2.2 direction) |
| `geometry` | local accuracy / completeness / Chamfer to the reference surface around the removed Gaussians (synthetic scene only) |
| `floater_precision`, `auroc_floater` | D1 sanity check against outcome labels (center > τ from the surface and visible on held-out views) |

`analyze_phase0.py` computes the paired gains `Gain = signal − mean(random)` per seed and
applies the go/no-go rule written at the top of that script (SNR ≥ 2 in every seed, gain of
consistent sign with |mean| ≥ 2·SE). With 3 seeds this is a pilot heuristic, not a
significance test. Gains of signals that are not measurable are not interpreted.

## Known issues to fix before Phase 1

- Image-space region mask is too wide: LR_img ≈ 0.90 even for random removal.
- 3D locality (Gain_LR param) is reported but not part of the decision rule.
- "Consistent" counts both signs; split into "expected direction" vs "opposite".
- Response is only measured after 1000 re-optimization steps; also measure right after removal.
