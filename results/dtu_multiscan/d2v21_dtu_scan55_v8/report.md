# D2 protocol v2.1 — dtu scan55 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan55_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 2.41%) | 32.94% | 0.713 (0.825) | 0.786 (0.756) |
| gradmag | 10.26% (random 2.41%) | 19.03% | 0.784 (0.825) | 0.779 (0.756) |
| visibility | 0.37% (random 2.41%) | 15.41% | 0.914 (0.825) | 0.764 (0.756) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.3 ± 0.3 | +1.2 ± 0.3 | no |
| gradmag | +7.9 ± 1.5 | +5.7 ± 1.2 | yes |
| visibility | +8.0 ± 3.6 | +5.7 ± 2.0 | yes |
| random | +8.2 ± 2.0 | +5.9 ± 1.8 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.024 ± 0.007 **E** | n/a | +5.327 ± 0.427 **E** | -0.089 ± 0.101 | +0.109 ± 0.011 **E** |
| gradmag | -0.003 ± 0.007 | n/a | -28.755 ± 4.261 **O** | +2.592 ± 3.571 | -0.270 ± 0.048 **O** |
| visibility | +0.037 ± 0.003 **E** | n/a | +6.642 ± 0.247 **E** | +7.978 ± 0.750 **E** | +0.109 ± 0.011 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.071 ± 0.033 **O** | +0.052 ± 0.022 **E** | -0.470 ± 0.214 **O** | -0.014 ± 0.274 | +0.068 ± 0.013 **E** |
| gradmag | -0.048 ± 0.026 **O** | +0.123 ± 0.017 **E** | +0.673 ± 1.163 | +4.067 ± 1.655 **E** | -0.119 ± 0.031 **O** |
| visibility | +0.073 ± 0.013 **E** | +0.051 ± 0.015 **E** | +0.374 ± 0.206 **E** | +4.245 ± 0.954 **E** | +0.064 ± 0.012 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.112 ± 0.055 **O** | +0.074 ± 0.022 **E** | -0.411 ± 0.203 **O** | +0.359 ± 0.390 | +0.060 ± 0.014 **E** |
| gradmag | -0.041 ± 0.022 | +0.091 ± 0.017 **E** | +0.700 ± 1.215 | +1.278 ± 1.100 | -0.097 ± 0.032 **O** |
| visibility | +0.089 ± 0.018 **E** | +0.038 ± 0.019 **E** | +0.451 ± 0.305 **E** | +4.288 ± 0.749 **E** | +0.059 ± 0.015 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: supported (localized and in the expected direction)