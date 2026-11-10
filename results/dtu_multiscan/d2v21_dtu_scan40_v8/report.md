# D2 protocol v2.1 — dtu scan40 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan40_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 3.00%) | 51.63% | 0.665 (0.786) | 0.703 (0.642) |
| gradmag | 10.66% (random 3.00%) | 33.34% | 0.744 (0.786) | 0.674 (0.642) |
| visibility | 0.52% (random 3.00%) | 19.48% | 0.887 (0.786) | 0.700 (0.642) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.1 ± 0.4 | +1.2 ± 0.3 | no |
| gradmag | +9.3 ± 1.8 | +8.1 ± 1.7 | yes |
| visibility | +10.4 ± 3.4 | +7.7 ± 2.2 | yes |
| random | +10.1 ± 1.5 | +8.0 ± 0.7 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.040 ± 0.008 **E** | n/a | +7.531 ± 0.659 **E** | -0.068 ± 0.205 | +0.244 ± 0.008 **E** |
| gradmag | -0.014 ± 0.013 | n/a | -17.166 ± 4.234 **O** | +8.695 ± 3.332 **E** | -0.151 ± 0.085 **O** |
| visibility | +0.063 ± 0.006 **E** | n/a | +8.168 ± 0.616 **E** | +6.472 ± 1.098 **E** | +0.244 ± 0.008 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.062 ± 0.040 | +0.100 ± 0.016 **E** | -1.371 ± 0.167 **O** | +0.183 ± 0.286 | +0.204 ± 0.021 **E** |
| gradmag | -0.049 ± 0.040 | +0.063 ± 0.021 **E** | +2.513 ± 1.437 **E** | +4.166 ± 1.760 **E** | -0.113 ± 0.064 **O** |
| visibility | +0.095 ± 0.012 **E** | +0.122 ± 0.010 **E** | -0.804 ± 0.183 **O** | +4.125 ± 0.692 **E** | +0.202 ± 0.018 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.121 ± 0.047 **O** | +0.090 ± 0.011 **E** | -0.899 ± 0.252 **O** | +0.431 ± 0.383 **E** | +0.190 ± 0.017 **E** |
| gradmag | -0.042 ± 0.041 | +0.054 ± 0.017 **E** | +1.579 ± 1.103 **E** | +3.409 ± 1.807 **E** | -0.088 ± 0.076 |
| visibility | +0.100 ± 0.017 **E** | +0.112 ± 0.009 **E** | -0.528 ± 0.248 **O** | +4.027 ± 0.484 **E** | +0.194 ± 0.020 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: supported (localized and in the expected direction)
- **visibility**: stated meaning not supported (direction opposite to expected)