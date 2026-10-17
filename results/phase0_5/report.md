# Phase 0.5 — D2 protocol v2

- Seeds: [0, 1, 2] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 2 random selections per seed · checkpoints and selections identical to Phase 0
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 1.40%) | 22.48% | nan (0.898) | 0.931 (0.901) |
| gradmag | 6.63% (random 1.40%) | 14.75% | 0.903 (0.898) | 0.913 (0.901) |
| visibility | 0.11% (random 1.40%) | 15.94% | 0.980 (0.898) | 0.896 (0.901) |

## Measurability (image SNR_in, weight mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +0.0 ± 0.0 | +0.0 ± 0.0 | no |
| gradmag | +6.0 ± 0.5 | +5.8 ± 0.3 | yes |
| visibility | +37.7 ± 29.1 | +43.2 ± 38.6 | yes |
| random | +7.0 ± 0.7 | +6.8 ± 0.7 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (weight mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | n/a | n/a | +1.604 ± 0.135 **E** | -0.267 ± 0.100 **O** | +0.399 ± 0.010 **E** |
| gradmag | +0.002 ± 0.004 | n/a | -6.914 ± 0.399 **O** | +0.202 ± 0.198 | -0.718 ± 0.119 **O** |
| visibility | +0.040 ± 0.003 **E** | n/a | +3.341 ± 1.385 **E** | +1.108 ± 1.793 | +0.367 ± 0.017 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (weight mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | n/a | +0.022 ± 0.021 | -0.045 ± 0.081 | -0.170 ± 0.084 **O** | +0.267 ± 0.004 **E** |
| gradmag | +0.000 ± 0.005 | +0.124 ± 0.008 **E** | +0.337 ± 0.269 **E** | -0.035 ± 0.141 | -0.233 ± 0.073 **O** |
| visibility | +0.070 ± 0.014 **E** | -0.001 ± 0.038 | +1.766 ± 1.573 | +1.204 ± 1.805 | +0.239 ± 0.012 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (weight mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | n/a | +0.045 ± 0.018 **E** | -0.061 ± 0.075 | -0.167 ± 0.124 **O** | +0.273 ± 0.024 **E** |
| gradmag | +0.005 ± 0.008 | +0.094 ± 0.018 **E** | +0.136 ± 0.114 **E** | -0.085 ± 0.237 | -0.184 ± 0.084 **O** |
| visibility | +0.082 ± 0.021 **E** | +0.000 ± 0.050 | +1.785 ± 1.539 **E** | +1.165 ± 1.658 | +0.250 ± 0.023 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: supported (localized and in the expected direction)