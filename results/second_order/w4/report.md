# W4 second-order signals: implementation and checkpoint export

**Status (27 October 2026):** all signals use Quỳnh Anh's official gradient-inconsistency v
(provider `quynhanh`, zero insufficient-view fallback) after merging main `02a1536`.
The three synthetic checkpoints were re-exported and pass the merged D1 pipeline
(`scripts/run_d1.py`, 30 metric rows, checkpoint hashes verified). Full suite at that
commit: **132 passed, 7 skipped**. Current evidence: `main_integration/`.
Earlier temporary-v exports and the Drive handoff package are kept only in the Git history
of this branch. GTX 1650 profiling was deferred.

| Field | Value |
|---|---|
| Owner / date | Phát / 23 October 2026 |
| Branch / base | `feature/phat-second-order` / latest fetched `origin/main`, `3aef207617946e65cda54170ae7b9b130e5be25a` |
| Source state | Local changes; source SHA256 snapshots in the environment JSON files |
| Data | [Shared checkpoint/data folder](https://drive.google.com/drive/folders/1XuBXSTEBo3JWrfLepIMWpzxbL6TTq_rW) |
| Scene / condition | Procedural synthetic, `v8_az360`; 8 training / 16 held-out views, 160 px |
| Seeds | Checkpoint seeds 0, 1, 2; separate probe-seed sweep 0, 1, 2 on checkpoint 0 |
| Backend / environment | PyTorch renderer, Python 3.12.7, torch 2.8.0+cu129, RTX 4060 Laptop 8 GiB |
| Definitions | `docs/research/Phat_second_order_and_d3.md`, A.2–A.5 |

## Source verification and output

Downloaded 39 files (79,006,083 bytes) to ignored `runs/drive_source/data/`.
`rclone check` found **39 matching files and zero differences**. The inventory with
Drive IDs/checksums and the transfer check are retained under `runs/drive_source/`.
The source package has no config file; the committed `results/phase0/config_used.yaml`
was independently checked by reproducing source metrics and labels.

| Seed | N | Base checkpoint SHA256 |
|---|---:|---|
| 0 | 27,733 | `95bb31a8830d1ca8ab1cadbc65b2c192014b0bbf533b3dd9a1c6b3fc4605c517` |
| 1 | 27,984 | `2483c9a07e7e20fd54834d289cf0afa7cee77b69dad0df2176af78e832d202e5` |
| 2 | 28,691 | `10bc6f76ed740546fede09d37a0e12661dfdc0a7d761422f0befe19367b857d1` |

All three checkpoints match the supplied manifests. Label `xyz` exactly matches
checkpoint row order; four baseline score arrays exactly match `signals.pt` and
remain unchanged after merging. Recomputed reference distances agree at
`rtol=1e-5, atol=1e-6`; provisional floater labels agree exactly. Maximum source-metric
differences: L1 1.41e-7, DSSIM 1.04e-6, PSNR 4.55e-5 dB. See
`main_integration/source_audit.csv` for individual values.

Each `runs/d1/synthetic/v8_az360/seed_{0,1,2}/scores.npz` contains:

- Preserved `random`, `opacity`, `gradmag`, `visibility`.
- Added `fisher`, `pup`, `curvature`, `gradient_inconsistency`, `s_geom`, `s_geom_ranksum`.
- Identity metadata, scene config/hash, per-signal S/epsilon/basis/provider settings.

Every score is finite float32 `(N,)`, in checkpoint order, with high = less reliable.
The source Drive package and local raw copy are unchanged. Original manifests retain
Huy's machine paths; `--checkpoint` supplies a relocated file while still checking
its original SHA256 and count. The actual arrays/checkpoints are ignored by Git.

## Method and numerical validation

W3 exact-image-Jacobian validation is in `../report.md`. W4 adds a diagonal all-parameter
trace estimator for Fisher, activated-scale 6x6 log-determinants for PUP, inverse
position-block trace for curvature, and product/rank-sum combinations with v.
These are documented paper adaptations, not reproductions of the authors' CUDA kernels.

The handoff uses **S=16**, PUP/curvature `eps=1e-8`, Fisher `eps=1e-12`, v `delta=1e-12`,
`n_min=2`, and PUP activated-scale basis. v uses contributing training views and
population variance of position loss gradients. Fewer than two eligible views
received v=1 (unknown) under the earlier temporary v: 666 / 800 / 638 Gaussians. With the
official v (provider `quynhanh`) the fallback count is zero. Settings per seed are in
`main_integration/seed_*_options.json`.

Sample/damping stability uses an **approximate S=256, probe-seed-0 reference**, not
an exact production Jacobian. Checkpoint 0 is fixed; three probe seeds and
S=4/16/64 are swept with eps=1e-8/1e-6/1e-4. The same probe prefixes are reused,
so the reference is correlated with trials. Results describe this scene/backend,
not a universal error bound or a hyperparameter selection on held-out labels.

Mean Spearman rank correlation against the reference at eps=1e-8:

| Signal | S=4 | S=16 | S=64 | S=16 top-20% overlap |
|---|---:|---:|---:|---:|
| fisher | 0.99182 | 0.99799 | 0.99945 | 0.97530 |
| pup | 0.99639 | 0.99938 | 0.99984 | 0.98762 |
| curvature | 0.99236 | 0.99852 | 0.99963 | 0.97782 |
| s_geom | 0.99240 | 0.99854 | 0.99963 | 0.97698 |
| s_geom_ranksum | 0.99323 | 0.99866 | 0.99965 | 0.97902 |

At S=16, changing eps from 1e-8 to 1e-4 gives reference correlations 0.99938→0.99924
for PUP, 0.99852→0.99852 for curvature and 0.99854→0.99803 for the product. Fisher's
separate epsilon stays fixed. v has no S/epsilon dependence; repeated CUDA passes
show tiny ranking differences (rho approximately 0.99997), consistent with renderer
floating-point reduction variability. Identical probe seeds do not guarantee
bitwise identical CUDA rendering. Detailed rows/plot are in `main_integration/stability/`.

The separate **constructed, untrained** 20-Gaussian fixture in `stability.csv/.png`
is a numerical diagnostic only. Its target images are 0.95 times its own renders;
it is not a failure-association or reconstruction-quality experiment. The generated
28,000-Gaussian profile in `profile.csv` is also separate from the trained results.

## Runtime and memory

Times measure the six-signal bundle after scene/checkpoint loading, with CUDA
synchronization. Allocated/reserved peaks are PyTorch process statistics, including
live scene/model tensors; they exclude driver/context allocations and other processes.

| Trained seed | N | S | Bundle seconds | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|---:|---:|
| 0 | 27,733 | 16 | 4.785 | 354.1 | 688.0 |
| 1 | 27,984 | 16 | 4.181 | 357.7 | 708.0 |
| 2 | 28,691 | 16 | 4.277 | 372.1 | 710.0 |

An initial generated-scene run exposed approximately 15 GiB of CUDA eigensolver
workspace for a single batch of 28k tiny matrices. Batching spectra in groups of
512 reduced the generated-fixture bundle peak to 319.4 MiB allocated (S=4, 1.861 s).
The regression test checks a 4096x6x6 CUDA spectrum and extra peak allocation below
1 GiB. These are **RTX 4060** measurements; the requested GTX 1650 benchmark is open.

## Initial association diagnostic

The source labels explicitly say `provisional=True`, `m_i_available=False`.
Floater prevalence is 8.398% / 9.044% / 8.599%. The following is a small AUROC sanity
check against supplied `f_i`, using existing `frecon.measure.auroc`. It does not
replace Tuấn Khải's D1 protocol and does not implement D3.

| Signal | Mean AUROC ± sample SD across 3 checkpoint seeds |
|---|---:|
| random | 0.5026 ± 0.0043 |
| opacity | 0.3486 ± 0.0204 |
| gradmag | 0.6536 ± 0.0104 |
| visibility | 0.3550 ± 0.0116 |
| fisher | 0.2984 ± 0.0087 |
| pup | 0.3069 ± 0.0179 |
| curvature | 0.3406 ± 0.0157 |
| gradient_inconsistency (official v) | 0.6507 ± 0.0250 |
| s_geom | 0.3934 ± 0.0027 |
| s_geom_ranksum | 0.4880 ± 0.0087 |

Values from `main_integration/scores_summary.csv`.

**Observation:** all arrays satisfy the contract; information/curvature/product
scores have AUROC below 0.5, despite high numerical rank stability on checkpoint 0.
v is comparable to GradMag in this small diagnostic.

**Interpretation:** numerical correctness/stability does not establish failure
prediction. Weakly observed Gaussians are uncertain, while the label requires
held-out visibility; this is one plausible reason for the direction of association,
not a verified explanation. Keep the declared orientation and test visibility
strata in W5. Do not flip scores after seeing labels or infer D2 faithfulness.

## Reproduction and regression checks

Run from the repository root after obtaining the shared source package at the paths
above. The authenticated `gdrive:` remote was used with the exact shared folder ID:

```powershell
rclone copy gdrive: runs/drive_source/data --drive-root-folder-id 1XuBXSTEBo3JWrfLepIMWpzxbL6TTq_rW --checksum
rclone check gdrive: runs/drive_source/data --drive-root-folder-id 1XuBXSTEBo3JWrfLepIMWpzxbL6TTq_rW --one-way
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
python -m pytest -q
python scripts/validate_second_order_checkpoints.py
python scripts/validate_second_order_signals.py --out_dir results/second_order/w4/main_integration/stability --run_dir runs/d1/synthetic/v8_az360/seed_0 --checkpoint runs/drive_source/data/seed_0/base.pt
```

For an existing export, the audit/export command requires explicit `--overwrite`
to replace the same six signals; existing baselines remain preserved. Standalone export:

```powershell
python scripts/export_second_order.py --run_dir runs/d1/synthetic/v8_az360/seed_0 --checkpoint runs/drive_source/data/seed_0/base.pt --config results/phase0/config_used.yaml --n_samples 16
python scripts/validate_second_order_signals.py --out_dir results/second_order/w4 --profile_n 28000
```

The full suite reports **49 passed, 14 skipped** in the stated environment.
`MKL_THREADING_LAYER=SEQUENTIAL`
avoids a native SciPy/Intel MKL failure encountered in new main's DTU loader tests
on this Windows installation. The current suite's skips cover unimplemented D1,
D3 and other-track tasks plus the optional gsplat backend; they are not completion evidence.

## Remaining acceptance and next action

- Re-evaluate fisher / pup / curvature against distance-based labels (d > 0.03, d > 0.1):
  the provisional `f_i` requires held-out contribution and opacity, which excludes weakly
  observed Gaussians by construction and may explain AUROC < 0.5.
- Report how many Gaussians have F = 0 (seen by no view); they tie at the top of the
  curvature ranking and would dominate a top-5% removal.
- Use one `n_samples` value (16) in both `REGISTRY` and the export, then write the signals to
  `signals.pt` so D2 can use them.
- D3 remains a future milestone.
