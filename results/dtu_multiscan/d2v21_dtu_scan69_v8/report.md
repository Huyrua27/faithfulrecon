# D2 protocol v2.1 — dtu scan69 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan69_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 4.16%) | 31.79% | 0.807 (0.759) | 0.712 (0.654) |
| gradmag | 8.67% (random 4.16%) | 20.02% | 0.785 (0.759) | 0.731 (0.654) |
| visibility | 0.19% (random 4.16%) | 11.86% | 0.896 (0.759) | 0.735 (0.654) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.6 ± 0.5 | +1.6 ± 0.4 | no |
| gradmag | +9.5 ± 3.6 | +6.8 ± 2.4 | yes |
| visibility | +5.4 ± 3.2 | +4.2 ± 2.2 | no |
| random | +8.9 ± 2.8 | +7.0 ± 2.5 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.073 ± 0.011 **E** | n/a | +4.814 ± 0.528 **E** | -0.244 ± 0.279 | +0.209 ± 0.014 **E** |
| gradmag | +0.024 ± 0.015 | n/a | -6.521 ± 2.314 **O** | +3.973 ± 3.984 | -0.168 ± 0.073 **O** |
| visibility (n.m.) | +0.081 ± 0.010 **E** | n/a | +5.279 ± 0.566 **E** | +5.354 ± 1.766 **E** | +0.209 ± 0.014 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.056 ± 0.011 **E** | +0.039 ± 0.012 **E** | -0.767 ± 0.405 **O** | +0.456 ± 0.430 **E** | +0.167 ± 0.012 **E** |
| gradmag | +0.003 ± 0.034 | +0.117 ± 0.017 **E** | +0.726 ± 0.659 | +0.740 ± 2.953 | -0.123 ± 0.077 **O** |
| visibility (n.m.) | +0.116 ± 0.029 **E** | +0.091 ± 0.023 **E** | -0.497 ± 0.472 | +4.948 ± 1.158 **E** | +0.164 ± 0.020 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.048 ± 0.062 | +0.052 ± 0.012 **E** | -0.626 ± 0.341 **O** | +0.592 ± 0.534 | +0.159 ± 0.022 **E** |
| gradmag | +0.026 ± 0.045 | +0.102 ± 0.016 **E** | +0.104 ± 0.947 | +0.599 ± 2.964 | -0.099 ± 0.068 **O** |
| visibility (n.m.) | +0.137 ± 0.047 **E** | +0.069 ± 0.017 **E** | -0.340 ± 0.530 | +4.724 ± 1.516 **E** | +0.160 ± 0.022 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: not measurable