# D2 protocol v2.1 — dtu scan63 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan63_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 1.04%) | 17.70% | 0.816 (0.897) | 0.864 (0.865) |
| gradmag | 3.29% (random 1.04%) | 10.28% | 0.891 (0.897) | 0.868 (0.865) |
| visibility | 0.45% (random 1.04%) | 10.84% | 0.984 (0.897) | 0.906 (0.865) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +0.5 ± 0.1 | +0.6 ± 0.0 | no |
| gradmag | +7.9 ± 1.7 | +6.3 ± 0.7 | yes |
| visibility | +25.7 ± 22.5 | +18.3 ± 16.0 | yes |
| random | +8.7 ± 1.2 | +7.4 ± 1.0 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.033 ± 0.003 **E** | n/a | +4.263 ± 0.223 **E** | -2.092 ± 0.214 **O** | +0.468 ± 0.050 **E** |
| gradmag | +0.008 ± 0.001 **E** | n/a | -5.008 ± 0.465 **O** | -0.432 ± 1.720 | -0.800 ± 0.243 **O** |
| visibility | +0.046 ± 0.002 **E** | n/a | +5.991 ± 1.748 **E** | +13.230 ± 10.348 **E** | +0.438 ± 0.052 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.059 ± 0.013 **O** | +0.100 ± 0.022 **E** | -0.286 ± 0.104 **O** | -2.129 ± 0.219 **O** | +0.347 ± 0.113 **E** |
| gradmag | -0.006 ± 0.007 | +0.140 ± 0.012 **E** | -0.313 ± 0.314 | +0.180 ± 1.585 | -0.636 ± 0.219 **O** |
| visibility | +0.073 ± 0.006 **E** | +0.020 ± 0.022 **E** | +1.313 ± 1.611 | +12.937 ± 10.452 **E** | +0.315 ± 0.125 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.081 ± 0.022 **O** | +0.108 ± 0.021 **E** | -0.280 ± 0.108 **O** | -2.258 ± 0.164 **O** | +0.297 ± 0.076 **E** |
| gradmag | -0.007 ± 0.010 | +0.119 ± 0.011 **E** | -0.315 ± 0.354 | -0.206 ± 1.273 | -0.531 ± 0.250 **O** |
| visibility | +0.087 ± 0.008 **E** | +0.023 ± 0.024 **E** | +1.131 ± 1.415 | +13.053 ± 10.684 **E** | +0.260 ± 0.103 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: supported (localized and in the expected direction)