# Phase 2 plan — D2 extensions

Phase 2 runs after the D2 protocol v2.1 results (`configs/d2v21_*.yaml`) and only as GPU time allows.
Items are in priority order. Each item lists the question, the change, the configuration and the
decision it informs. Owner: Huy unless noted.

## 0. Statistical requirements (apply to every item)

- **At least 6 paired observations per comparison.** A two-sided paired Wilcoxon signed-rank test with
  n = 5 cannot reach p < 0.05 (minimum p = 2/2⁵ = 0.0625), even when all differences have the same
  sign. Unit of observation = (scene, view condition, seed). Current state: v8_az360 has seeds 0–5
  (6 pairs); v8_az180 has seeds 0–2 (3 pairs). Pooling conditions uses condition as a block.
- Paired Wilcoxon signal vs. mean of matched random selections, Holm–Bonferroni across signals,
  bootstrap 95% CI of the gain. The sign-consistency rule of the pilot is kept only as a descriptive check.
- Primary step T = 1000; T = 0 reported alongside (immediate effect).

## 1. Selection ratio sweep — K ∈ {1, 2, 5, 10} %

- **Question.** Is a signal's D2 outcome specific to K = 5 %? Floaters are ~8.5 % of the Gaussians in
  v8_az360. An imperfect signal may select mostly floaters at small K and spill onto valid surface at
  larger K, which would flip the sign of Gain_Dir(Comp).
- **Change.** `intervention.frac` becomes a list; selections and random controls per K; regions,
  locality and direction per K. Report the gain as a curve over K.
- **Config.** v8_az360, seeds 0–5, signals {gradmag, visibility, opacity} + second-order signals when
  available, n_random = 5, removal, T ∈ {0, 200, 1000}.
- **Cost.** ≈ (2 null + 3 signals + 5 random) per K per seed: 3 extra K values × 6 seeds × 8 runs ≈ 144
  re-optimizations ≈ 8 h on a GTX 1650 → run on the lab/Colab GPU, or start with K ∈ {1, 2} % only.
- **Decision.** If gradmag is still "opposite" in completeness at K = 1–2 %, the Phase 0.5 finding does
  not depend on K. If it flips to "expected", report the finding as K-dependent.

## 2. Density-adaptive neighborhood radius

- **Question.** Does the fixed radius r = 0.05 (scene units) bias LR_3D and the local accuracy and
  completeness terms? A fixed radius is too wide where primitives are dense and too narrow where they are sparse.
- **Change.** Replace r by r_i = c · d_k(i), the mean distance from removed primitive i to its k nearest
  primitives (k = 8, c ∈ {1, 2, 4}). The neighborhood is the union of balls of radius r_i.
- **Config.** Re-measure only (no new re-optimization): `scripts/d2_v2.py` recomputes metrics from the
  cached `run_*.pt`. Report fixed-r and adaptive-r side by side.
- **Decision.** Keep the adaptive radius as primary if the signal rankings agree and its gains have
  smaller seed-to-seed variance; otherwise keep r = 0.05 and report the adaptive radius as an ablation.

## 3. Additional interventions

- Opacity scaling `α ← γα`, γ ∈ {0.5, 0.1}, and position perturbation `μ ← μ + η`,
  η ~ N(0, σ²I), with σ relative to the primitive scale. Same measurements as for removal.
- Question: does the outcome of a signal depend on the type of intervention?

## 4. More signals and conditions

- Second-order signals (Phát: fisher, pup, curvature, s_geom) and gradient inconsistency (Quỳnh Anh) in D2.
- Conditions: v8_az180 to 6 seeds; one sparser condition (e.g. v4_az360); a mesh-rendered scene once the
  real-scene choice (DTU / NeRF-Synthetic) is made.

## 5. Not planned for Phase 2

- First-order (influence-function) prediction of the response: optional, only if D2 results are stable.
