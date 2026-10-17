# D2 protocol v2.1 — condition v8_az360

- 8 training views over 360° · base models and signals from runs/phase0
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 1.34%) | 22.63% | 0.861 (0.889) | 0.935 (0.903) |
| gradmag | 6.70% (random 1.34%) | 14.83% | 0.901 (0.889) | 0.912 (0.903) |
| visibility | 0.11% (random 1.34%) | 16.12% | 0.939 (0.889) | 0.888 (0.903) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.1 ± 0.1 | +1.2 ± 0.1 | no |
| gradmag | +5.9 ± 0.4 | +5.5 ± 0.5 | yes |
| visibility | +8.3 ± 6.3 | +8.7 ± 6.5 | yes |
| random | +6.8 ± 0.8 | +5.8 ± 0.7 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.023 ± 0.003 **E** | n/a | +1.499 ± 0.133 **E** | -0.317 ± 0.042 **O** | +0.391 ± 0.018 **E** |
| gradmag | +0.002 ± 0.003 | n/a | -6.778 ± 0.444 **O** | +0.141 ± 0.097 **E** | -0.747 ± 0.096 **O** |
| visibility | +0.041 ± 0.001 **E** | n/a | +2.916 ± 1.287 **E** | +0.691 ± 1.259 | +0.354 ± 0.019 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.048 ± 0.008 **O** | +0.020 ± 0.011 **E** | -0.062 ± 0.103 | -0.202 ± 0.033 **O** | +0.247 ± 0.023 **E** |
| gradmag | +0.004 ± 0.005 | +0.118 ± 0.013 **E** | +0.160 ± 0.262 | -0.086 ± 0.162 | -0.283 ± 0.080 **O** |
| visibility | +0.036 ± 0.037 | +0.005 ± 0.038 | +1.413 ± 1.360 **E** | +0.780 ± 1.281 | +0.213 ± 0.029 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.028 ± 0.012 **O** | +0.045 ± 0.013 **E** | -0.023 ± 0.111 | -0.145 ± 0.130 | +0.231 ± 0.037 **E** |
| gradmag | +0.012 ± 0.005 **E** | +0.093 ± 0.013 **E** | -0.023 ± 0.228 | -0.015 ± 0.140 | -0.239 ± 0.074 **O** |
| visibility | +0.050 ± 0.039 | +0.010 ± 0.046 | +1.487 ± 1.403 **E** | +0.829 ± 1.297 | +0.195 ± 0.042 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: no evidence either way