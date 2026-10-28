# D2 protocol v2.1 — dtu scan83 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan83_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 1.08%) | 15.30% | 0.958 (0.889) | 0.934 (0.904) |
| gradmag | 3.60% (random 1.08%) | 9.37% | 0.909 (0.889) | 0.907 (0.904) |
| visibility | 0.03% (random 1.08%) | 8.26% | 0.986 (0.889) | 0.933 (0.904) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.9 ± 0.5 | +1.9 ± 0.6 | no |
| gradmag | +5.4 ± 1.1 | +4.8 ± 1.4 | yes |
| visibility | +12.6 ± 5.6 | +11.0 ± 6.2 | yes |
| random | +6.0 ± 0.9 | +5.4 ± 1.4 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.043 ± 0.008 **E** | n/a | +3.242 ± 0.114 **E** | -0.176 ± 0.024 **O** | +0.256 ± 0.018 **E** |
| gradmag | +0.014 ± 0.008 **E** | n/a | -5.222 ± 0.807 **O** | -0.320 ± 0.125 **O** | -0.252 ± 0.084 **O** |
| visibility | +0.050 ± 0.009 **E** | n/a | +3.492 ± 0.155 **E** | +1.265 ± 0.512 **E** | +0.234 ± 0.037 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.063 ± 0.010 **E** | +0.103 ± 0.021 **E** | -0.033 ± 0.093 | -0.257 ± 0.044 **O** | +0.218 ± 0.053 **E** |
| gradmag | +0.015 ± 0.013 | +0.104 ± 0.023 **E** | +0.498 ± 0.386 | -0.287 ± 0.149 **O** | -0.217 ± 0.103 **O** |
| visibility | +0.082 ± 0.011 **E** | +0.114 ± 0.024 **E** | +0.182 ± 0.130 **E** | +1.157 ± 0.498 **E** | +0.176 ± 0.060 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.070 ± 0.009 **E** | +0.121 ± 0.029 **E** | -0.017 ± 0.133 | -0.299 ± 0.034 **O** | +0.169 ± 0.066 **E** |
| gradmag | +0.021 ± 0.015 | +0.091 ± 0.020 **E** | +0.507 ± 0.345 **E** | -0.269 ± 0.151 **O** | -0.234 ± 0.104 **O** |
| visibility | +0.097 ± 0.012 **E** | +0.101 ± 0.018 **E** | +0.177 ± 0.164 | +1.104 ± 0.460 **E** | +0.115 ± 0.089 |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: supported (localized and in the expected direction)