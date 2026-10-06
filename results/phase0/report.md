# Phase 0 report — D2 proof of concept

- Run dir: `runs\phase0`
- Seeds: [0, 1, 2]
- Scene: synthetic (8 training views, 16 held-out views, 160 px)
- K = 5%, removal, re-optimization 1000 steps, 2 random selections per seed

## Base models

| seed | #Gaussians | held-out PSNR | floater fraction |
|---|---|---|---|
| 0 | 27733 | 27.51 | 8.398% |
| 1 | 27984 | 27.47 | 9.044% |
| 2 | 28691 | 27.50 | 8.599% |

Noise floor (null_b vs null_a) held-out ΔPSNR per seed: -0.003, -0.008, -0.005

## D2.1 / D2.3 — locality and matched contrast

Values: mean ± std over seeds. Gains are paired differences vs the mean of the random selections of the same seed. SNR_in = response inside the selected region / noise floor inside the same region.

| signal | SNR_in | SNR_in (random) | LR_img | LR_img (random) | Gain_LR (image) | Gain_LR (param) |
|---|---|---|---|---|---|---|
| opacity | 1.10 ± 0.05 | 2.69 ± 0.18 | 0.9303 ± 0.0046 | 0.9013 ± 0.0062 | 0.0290 ± 0.0077 | 0.0460 ± 0.0163 |
| gradmag | 4.71 ± 0.38 | 2.69 ± 0.18 | 0.9127 ± 0.0044 | 0.9013 ± 0.0062 | 0.0113 ± 0.0071 | 0.0943 ± 0.0187 |
| visibility | 1.77 ± 0.57 | 2.69 ± 0.18 | 0.8943 ± 0.0115 | 0.9013 ± 0.0062 | -0.0070 ± 0.0170 | 0.0021 ± 0.0474 |

## D2.2 / D2.3 — direction

Dir = −(change in error after removal); positive = removal made the reconstruction better (or hurt it less). Gain_Dir = Dir(signal) − Dir(random).

| signal | Dir (1−SSIM) | |noise| (1−SSIM) | Gain_Dir (1−SSIM) | Gain_Dir (L1) | Dir (local CD) | |noise| (local CD) | Gain_Dir (local CD) |
|---|---|---|---|---|---|---|---|
| opacity | -0.00001 ± 0.00004 | 0.00005 ± 0.00003 | -0.00005 ± 0.00016 | -0.00001 ± 0.00004 | 0.00002 ± 0.00005 | 0.00006 ± 0.00008 | 0.00005 ± 0.00004 |
| gradmag | 0.00011 ± 0.00021 | 0.00005 ± 0.00003 | 0.00007 ± 0.00007 | 0.00002 ± 0.00005 | -0.00016 ± 0.00007 | 0.00005 ± 0.00007 | -0.00013 ± 0.00007 |
| visibility | 0.00182 ± 0.00160 | 0.00005 ± 0.00003 | 0.00178 ± 0.00155 | 0.00024 ± 0.00022 | 0.00066 ± 0.00082 | 0.00005 ± 0.00009 | 0.00069 ± 0.00081 |

## D1 sanity — floater labels (not a Phase 0 criterion)

| signal | AUROC vs floater | floater precision@K | precision@K (random) |
|---|---|---|---|
| opacity | 0.349 ± 0.020 | 0.000 ± 0.000 | 0.083 ± 0.005 |
| gradmag | 0.654 ± 0.010 | 0.108 ± 0.009 | 0.083 ± 0.005 |
| visibility | 0.355 ± 0.012 | 0.027 ± 0.032 | 0.083 ± 0.005 |

## Go / no-go for D2

Rule: measurable = SNR_in ≥ 2.0 in every seed; consistent = Gain has the same sign in every seed and |mean| ≥ 2·SE. GO if at least one signal is measurable and consistent.

| signal | measurable | consistent Gain_LR (image) | consistent Gain_Dir (1−SSIM) | consistent Gain_Dir (local CD) |
|---|---|---|---|---|
| opacity | no | – | – | – |
| gradmag | yes | yes | no | yes |
| visibility | no | – | – | – |

Random control measurable (SNR_in ≥ 2.0 in every seed): **yes** (2.80, 2.47, 2.79)

**Verdict: GO — at least one signal gives a measurable and consistent D2 response.**
