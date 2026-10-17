# Validation — Quỳnh Anh signals

Date: 2026-10-16 (Asia/Bangkok).

## Runtime

- Python: D:/Programs/anaconda3/envs/chess-rl/python.exe
- PyTorch: 2.4.1+cpu; CPU, 4 threads during export.
- NumPy 1.26.4; SciPy 1.16.2; scikit-learn 1.7.2.
- Renderer: torch. Recompute config: results/phase0/config_used.yaml.

## Commands actually completed

```powershell
& './scripts/reproduce_quynhanh.ps1' -Python 'D:/Programs/anaconda3/envs/chess-rl/python.exe'
& 'D:/Programs/anaconda3/envs/chess-rl/python.exe' -m pytest -q --basetemp runs/pytest_quynhanh_20261016_b
git diff --check
```

Reproduce completed all three cached exports, three checkpoint exports and the analysis script.
Each checkpoint export (training image generation, five scores, CSVs and 15 figures) took about 11 seconds on this machine.
pytest: **28 passed, 19 skipped**, 4.02 seconds. Skips are pre-existing unimplemented benchmark/evaluation/second-order/D3 tests; the new gradient-inconsistency test is enabled.
git diff --check passed; Git only reported LF/CRLF conversion warnings for the two edited documentation files.

The pytest basetemp path above was fresh when used. For a new sandbox run choose another unused path or let pytest use the ordinary system temp outside this sandbox.

## Artifact integrity checks actually completed

| Seed | N | Recomputed score/CSV rows (5 signals) | Top1 / Top5 / Top10 count per signal |
|---|---:|---:|---|
| 0 | 27733 | 138665 | 277 / 1387 / 2773 |
| 1 | 27984 | 139920 | 280 / 1399 / 2798 |
| 2 | 28691 | 143455 | 287 / 1435 / 2869 |

- Total recomputed CSV data rows: **422040**, excluding the three headers.
- Every CSV row checked against its score, Gaussian row ID, signal name and seed.
- All recomputed arrays verified shape (N,), float32, finite.
- Rankings verified as permutations of 0..N-1 with nonincreasing scores.
- Every Top1/5/10 ID array verified against the corresponding ranking prefix and expected size.
- Every checkpoint hash matches the supplied manifest and exported score metadata.
- All implementation file hashes recorded by the exporter match the code used for the final run.
- Cached exports verified exactly equal to the four supplied score arrays for every seed.
- **45 recomputed PNGs** (five signals × three K × three seeds) and **36 cached PNGs** (four signals × three K × three seeds).
- Inspected representative recomputed GI Top5 and opacity Top1 PNGs for axes, title, bounds, selected count, color and clipping.

## Limits

These checks establish implementation/output consistency for the supplied checkpoints. They do not establish geometric reliability or removal benefit. GI D2 has not been run; provisional f_i and unavailable m_i remain unchanged. Original CPU/CUDA runtime drift is reported in QuynhAnh_signal_audit.md.
