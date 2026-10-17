# D2 protocol v2.1 — condition v8_az180

- 8 training views over 180° · base models and signals from runs/v8_az180
- Seeds: [0, 1, 2] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 1.28%) | 21.55% | 0.857 (0.915) | 0.930 (0.903) |
| gradmag | 6.29% (random 1.28%) | 14.87% | 0.887 (0.915) | 0.900 (0.903) |
| visibility | 0.07% (random 1.28%) | 16.40% | 0.875 (0.915) | 0.864 (0.903) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +0.9 ± 0.1 | +1.1 ± 0.1 | no |
| gradmag | +5.6 ± 0.5 | +5.6 ± 0.1 | yes |
| visibility | +2.5 ± 0.4 | +3.2 ± 0.9 | yes |
| random | +9.5 ± 1.4 | +8.9 ± 0.3 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.020 ± 0.000 **E** | n/a | +1.318 ± 0.118 **E** | -0.282 ± 0.015 **O** | +0.407 ± 0.033 **E** |
| gradmag | -0.003 ± 0.002 **O** | n/a | -6.763 ± 0.298 **O** | +0.069 ± 0.174 | -0.702 ± 0.179 **O** |
| visibility | +0.035 ± 0.000 **E** | n/a | +1.446 ± 0.144 **E** | +1.199 ± 1.967 | +0.370 ± 0.029 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.075 ± 0.003 **O** | +0.043 ± 0.002 **E** | -0.074 ± 0.049 **O** | -0.201 ± 0.088 **O** | +0.307 ± 0.046 **E** |
| gradmag | -0.028 ± 0.003 **O** | +0.144 ± 0.013 **E** | -0.107 ± 0.096 | -0.086 ± 0.105 | -0.291 ± 0.151 **O** |
| visibility | -0.049 ± 0.016 **O** | -0.011 ± 0.056 | +0.074 ± 0.073 | +1.226 ± 2.027 | +0.274 ± 0.048 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.058 ± 0.004 **O** | +0.074 ± 0.015 **E** | -0.023 ± 0.063 | -0.230 ± 0.046 **O** | +0.289 ± 0.036 **E** |
| gradmag | -0.028 ± 0.006 **O** | +0.112 ± 0.014 **E** | -0.254 ± 0.156 **O** | -0.164 ± 0.148 | -0.288 ± 0.143 **O** |
| visibility | -0.040 ± 0.033 **O** | +0.013 ± 0.070 | +0.136 ± 0.031 **E** | +1.199 ± 2.056 | +0.239 ± 0.063 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: no evidence either way