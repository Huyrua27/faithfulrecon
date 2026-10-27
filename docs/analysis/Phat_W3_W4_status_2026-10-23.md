# Phát: W3–W4 delivery status and next objectives

**27 October handoff update:** see `docs/analysis/Phat_W4_handoff_2026-10-27.md`
for the portable Drive package, completed D1 consumer check, and user-deferred
GTX 1650 task. The status below is the 23 October snapshot.

Updated 23 October 2026 against `origin/main` at `3aef207` on
`feature/phat-second-order`. The earlier 13 October plan is retained as a dated
assessment; this document is the current implementation and handoff status.

## Last week: W3, 12–18 October

The first objective was a numerically validated rendered-image Gauss–Newton estimator.
This is implemented for both position 3x3 and position/log-scale 6x6 blocks, with
normalization, deterministic probe streams, caching and preservation of model state.
The exact-Jacobian comparison and S=4/64/1024 convergence study are complete.
Mean relative error at S=1024 is 4.084% / 4.145%, respectively. The experiment uses
a tiny scene; it does not establish production signal quality.

Evidence: `results/second_order/report.md`, `tests/test_second_order.py`, research A.1.
Formal PR/reviewer acceptance remains outstanding.

## This week: W4, 19–25 October

| Deliverable | Current evidence | Status |
|---|---|---|
| Latest main and isolated work | GitHub fetch succeeded; feature branch fast-forwarded to `3aef207` without losing staged W3 work | Done locally |
| Fisher / PUP / curvature | Correct orientation, explicit parameter basis/damping, finite float32 `(N,)` outputs; formula and registry tests | Done |
| Candidate `s_geom` | u, temporary v, product, rank-sum; explicit unknown-view semantics and provider metadata | Done locally; coordinate v with Quỳnh Anh |
| Shared data audit | 39 Drive files checksum-verified; all three base hashes/N, label row coordinates, baseline arrays and source metrics verified | Done |
| `scores.npz` export | Six new arrays plus four preserved baselines for seeds 0/1/2 under `runs/d1/synthetic/v8_az360/` | Done locally |
| S/epsilon sensitivity | Constructed fixture and trained seed-0 ranking study, three probe seeds; reports under `results/second_order/w4/` | See report for measured scope |
| Runtime/memory | Three trained runs at S=16: 4.18–4.78 s, 354–372 MiB peak allocated, RTX 4060 Laptop | Local measurement done; GTX 1650 still required |
| Research and reproducible evidence | Formula/paper adaptations, experiment scripts, source hashes, reports and commands | Done locally |
| Regression suite | 49 passed, 14 skipped; skipped work belongs to incomplete D1/D3/other tracks and optional backend | Done in documented environment |
| Team delivery | Local artifacts prepared; no upload, PR or teammate notification performed | Pending review/distribution |

All results use training views to construct signals. The supplied `f_i` labels are
provisional and `m_i` is unavailable. The initial association diagnostic is negative
for the information/curvature/product scores, so implementation completion must not
be presented as evidence that these signals outperform baselines.

## Next objective: close the W4 handoff

1. Review `results/second_order/w4/report.md` and per-signal options with Huy.
   Agree that the current signal convention and temporary v are suitable for D1/D2.
2. Give Tuấn Khải the three score archives together with checkpoint hashes, source
   configuration, and provider/options metadata. Keep the provisional-label warning.
   Have his loader ingest them without edits and check a matching scene/condition/seed/hash.
3. Run the same trained-checkpoint benchmark on the assignment's GTX 1650. Current
   measurements are from an RTX 4060 and cannot substitute for that result.
4. Prepare the focused review package: source, meaningful tests, method notes and
   small evidence only. Checkpoints and per-Gaussian arrays remain ignored in `runs/`.

These are the remaining W4 acceptance steps. Local implementation and export are
ready for them; external acceptance and GTX 1650 access are not assumed complete.

## W5: 26 October–1 November

First experimental objective: explain the negative provisional-floater association
without tuning score direction to labels. Compare u, v, product and rank-sum using
the recorded direction. Predeclare all-Gaussian and visibility-qualified strata;
uncertain invisible primitives can be negatives under the visibility-gated `f_i`.
Report their counts separately rather than dropping them silently.

Then compare activated-scale vs log-scale PUP and the selected S/epsilon settings.
Use rank agreement and top-K overlap as numerical diagnostics; use D1 only with the
same checkpoint and label protocol. Select settings on declared development data,
freeze them, and hand frozen signal definitions to Huy for matched D2 interventions.
Do not infer D2 direction or locality from D1 association.

## W6–W7: 2–15 November

Implement D3 in this order:

1. Feature-gradient held-out error attribution with a tiny explicit-weight oracle.
   Detach pixel errors, use a one-channel background and an exact blending denominator;
   define how zero-contribution Gaussians are excluded.
2. Gaussian-level Spearman and failure-label AUROC with matching random baselines.
3. Fixed pixel-map normalization, eligibility mask and sparsification/AUSE. Verify
   an oracle map gives zero AUSE; handle score ties consistently.
4. One-command `run_d3.py`, CSV/figures and three-seed synthetic validation. Keep
   score construction on training views and any calibration split separate from test.

W8 joins D1–D2–D3 by checkpoint hash, signal parameters, seed, view condition,
renderer and protocol. D3 and that joined table are still future work.
