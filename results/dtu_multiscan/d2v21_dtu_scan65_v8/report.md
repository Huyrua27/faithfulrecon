# D2 protocol v2.1 — dtu scan65 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan65_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 2.39%) | 21.46% | 0.770 (0.833) | 0.798 (0.832) |
| gradmag | 6.74% (random 2.39%) | 17.59% | 0.871 (0.833) | 0.849 (0.832) |
| visibility | 0.23% (random 2.39%) | 9.77% | 0.922 (0.833) | 0.809 (0.832) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +0.5 ± 0.2 | +0.8 ± 0.2 | no |
| gradmag | +9.5 ± 2.5 | +7.8 ± 0.9 | yes |
| visibility | +4.9 ± 3.9 | +3.9 ± 1.6 | no |
| random | +10.4 ± 2.2 | +8.7 ± 1.5 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.067 ± 0.010 **E** | n/a | +2.170 ± 0.159 **E** | +0.054 ± 0.339 | +0.410 ± 0.050 **E** |
| gradmag | +0.029 ± 0.011 **E** | n/a | -2.559 ± 1.551 | +9.013 ± 11.398 | -0.515 ± 0.192 **O** |
| visibility (n.m.) | +0.072 ± 0.012 **E** | n/a | +2.261 ± 0.155 **E** | +11.872 ± 4.200 **E** | +0.410 ± 0.050 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.049 ± 0.018 **O** | +0.080 ± 0.013 **E** | -0.277 ± 0.081 **O** | +0.138 ± 0.651 | +0.275 ± 0.051 **E** |
| gradmag | +0.028 ± 0.019 **E** | +0.116 ± 0.013 **E** | +0.973 ± 0.245 **E** | +3.929 ± 8.193 | -0.307 ± 0.165 **O** |
| visibility (n.m.) | +0.093 ± 0.023 **E** | +0.216 ± 0.026 **E** | -0.121 ± 0.105 | +7.140 ± 2.102 **E** | +0.268 ± 0.049 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.064 ± 0.022 **O** | +0.069 ± 0.015 **E** | -0.195 ± 0.073 **O** | +0.345 ± 0.381 | +0.247 ± 0.032 **E** |
| gradmag | +0.038 ± 0.023 **E** | +0.103 ± 0.009 **E** | +0.736 ± 0.366 **E** | +2.892 ± 6.627 | -0.329 ± 0.146 **O** |
| visibility (n.m.) | +0.089 ± 0.029 **E** | +0.177 ± 0.028 **E** | -0.101 ± 0.109 | +7.098 ± 2.300 **E** | +0.238 ± 0.032 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: not measurable