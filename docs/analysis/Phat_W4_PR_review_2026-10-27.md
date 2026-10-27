# Final W4 evaluation before PR — 27 October 2026

**Decision:** ready for Huy's W4 implementation review, with the GTX 1650 benchmark
explicitly deferred by the user. No unresolved implementation blocker was found
after the main-integration fixes below. This is not scientific approval of the
signals, completion of D3, or permission to merge without Huy's review.

## Reviewed scope and current base

The feature branch now includes latest fetched main `02a1536`, which merged
Quỳnh Anh's signals and Tuấn Khải's D1. The PR adds Phát's rendered-image block
estimator, Fisher/PUP/curvature/product/rank-sum signals, safe export/packaging,
tests, research notes and small experiment artifacts. The diff against main does
not change the renderer/core, shared data format, or teammates' implementations.
The W3 estimator foundation is required by the W4 signals and is part of this PR.

Reviewed numerical definitions, parameter basis/chain rule, normalization and score
orientation; deterministic probe generation and cache assumptions; zero-information
handling, damping and CUDA eigensolver workspace; model/gradient preservation;
checkpoint/config identity, array order and baseline preservation; archive relocation
and checksums; integration with official v, standard registry and merged D1.

## Findings resolved before PR

1. **Official-v integration changed the low-view convention.** The temporary v
   assigned 1 with fewer than two eligible views, while the newly merged official
   implementation assigns 0. Delegation already selected that implementation, but
   metadata only named its provider and the old tests assumed the stub.
   The wrapper now records provider, fallback, unsupported count, effective delta,
   thresholds and loss weight; custom thresholds also delegate to the official
   implementation. A new regression tests the official zero-fallback branch with
   a custom visibility threshold. The historical fallback test explicitly isolates
   `NotImplementedError`. No teammate source was edited to resolve this.
2. **Package instructions became stale after main advanced.** The builder's README
   no longer describes main's D1 as a stub or hardcodes the temporary-v convention.
   A distinct official-v package was generated; the original package is preserved
   with its original metadata and evidence rather than overwritten or relabeled.

The earlier CUDA eigensolver workspace issue is already fixed by 512-matrix batches
and a meaningful GPU-memory regression. The tiny exact-Jacobian checks remain active.

## Final validation

| Check | Result |
|---|---|
| Full suite on integrated branch | **132 passed, 7 skipped**, 26.86 s |
| Skip scope | Four other-track geometry-label placeholders, two future D3 tests, unavailable optional gsplat backend |
| Checkpoint/data audit | Three SHA256/N matches; exact label xyz/order and baseline preservation; source metrics/labels reproduced |
| Fresh export | All six new arrays finite float32 `(N,)` for all three seeds; four baselines preserved |
| Provider | `quynhanh`, population moments, delta=effective_delta=1e-12, n_min=2, low-view fallback=0 |
| Unsupported counts | 666 / 800 / 638 Gaussians for seeds 0 / 1 / 2 |
| Merged D1 CLI | Exit 0; **30 rows**, ten signals × three seeds; R=100 matched random draws |
| Consumer warnings | Only provisional labels and unavailable m_i; checkpoint bytes verified |
| Consumer metric agreement | AUROC matches independent handoff diagnostic exactly for all 30 rows |
| Package validation | 17 payload files + manifest; file-set/SHA256 checks pass; remote download comparison: 1 match, 0 differences |

Source audit, settings, summaries, consumer CSV, environment/source fingerprints and
delivery receipt are in `results/second_order/w4/main_integration/`.

The refreshed trained seed-0 S/epsilon study uses the official v, three probe seeds,
S=4/16/64 and eps=1e-8/1e-6/1e-4 against an approximate S=256 reference. At S=16,
eps=1e-8, mean rank correlations are 0.99799 Fisher, 0.99938 PUP, 0.99852 curvature,
0.99854 product and 0.99857 rank-sum. This supports the current numerical budget on
this scene, not a universal accuracy bound or validity of the scientific hypothesis.

## Observation and interpretation

Mean AUROC against **provisional** f_i over three checkpoints:

| Signal | AUROC mean ± sample SD |
|---|---:|
| random | 0.5026 ± 0.0043 |
| legacy GradMag baseline | 0.6536 ± 0.0104 |
| fisher | 0.2984 ± 0.0087 |
| pup | 0.3069 ± 0.0179 |
| curvature | 0.3406 ± 0.0157 |
| official gradient_inconsistency | 0.6507 ± 0.0250 |
| s_geom | 0.3934 ± 0.0027 |
| s_geom_ranksum | 0.4880 ± 0.0087 |

**Observation:** the implementation/contract checks pass, but the information,
curvature and product scores remain below random in this provisional diagnostic.
**Interpretation:** numerical correctness and stable ranks do not establish failure
prediction or D2 faithfulness. Score orientation was not reversed after observing
labels. Visibility-stratified analysis and controlled D2 experiments remain W5 work.
The retained baseline arrays are the supplied Phase 0 arrays, not regenerated with
the newer baseline implementations on main.

## Current handoff and reproduction

Use the [official-v package](https://drive.google.com/file/d/1GRq34RBI9gDKDD0Kf3aM6d5mdXfZNtU6/view?usp=drivesdk)
from the [team handoff folder](https://drive.google.com/drive/folders/10Rz779ch6FJecUD9kKCFrWpAdgPoJjew).
It is 17,444,643 bytes, SHA256
`4b78bafca0453a298637dd72d5ad88c095a197701f2c95e5e3fcd4f6966c27f7`.
The earlier temporary-v ZIP is historical and must not be pooled as the same signal
definition. Both packages preserve the original checkpoint hashes and baseline arrays.

From the feature branch with the project dependencies installed:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
python -m pytest -q
python scripts/package_second_order_handoff.py --verify runs/handoff/w4-second-order-2026-10-27-official-v.zip
python -m zipfile -e runs/handoff/w4-second-order-2026-10-27-official-v.zip runs/handoff/received-official-v
python scripts/run_d1.py --scene synthetic --condition v8_az360 --runs_root runs/handoff/received-official-v/d1 --out_dir runs/pr_review/main_d1 --n_random 100
```

To recreate the exports/stability study from the original shared source:

```powershell
python scripts/validate_second_order_checkpoints.py --out_root runs/pr_review/d1 --results results/second_order/w4/main_integration
python scripts/validate_second_order_signals.py --out_dir results/second_order/w4/main_integration/stability --run_dir runs/pr_review/d1/synthetic/v8_az360/seed_0 --checkpoint runs/drive_source/data/seed_0/base.pt
```

Use `--overwrite` only when intentionally replacing an existing export. The environment
used Anaconda Python 3.12.7, torch 2.8.0+cu129 and RTX 4060 Laptop. The MKL setting
avoids the documented Windows SciPy/MKL issue. Numeric bitwise equality across
CUDA renderer executions or platforms is not claimed.

## Remaining acceptance / future scope

- GTX 1650 runtime/memory remains a user-deferred task; RTX 4060 evidence is labeled.
- Huy reviews formulas, scope, metadata and the official-v insufficient-view policy.
- Official geometry labels/m_i are unavailable; this package uses provisional f_i.
- D3, its documentation and the combined D1–D2–D3 table are not implemented here.
  The task checklist therefore has three fully checked items; whole-assignment
  documentation remains partial despite completed W3/W4 method and reproduction notes.
