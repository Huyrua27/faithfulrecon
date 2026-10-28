# D2 protocol v2.1 — dtu scan24 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan24_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 3.12%) | 47.75% | 0.700 (0.769) | 0.710 (0.665) |
| gradmag | 11.64% (random 3.12%) | 32.49% | 0.752 (0.769) | 0.706 (0.665) |
| visibility | 0.34% (random 3.12%) | 15.17% | 0.897 (0.769) | 0.716 (0.665) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +0.8 ± 0.1 | +1.0 ± 0.2 | no |
| gradmag | +7.8 ± 1.2 | +6.9 ± 0.7 | yes |
| visibility | +8.2 ± 4.0 | +7.4 ± 3.3 | yes |
| random | +8.6 ± 1.4 | +7.4 ± 1.2 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.058 ± 0.004 **E** | n/a | +4.745 ± 0.414 **E** | -0.959 ± 0.281 **O** | +0.212 ± 0.018 **E** |
| gradmag | +0.015 ± 0.007 **E** | n/a | -20.036 ± 2.547 **O** | +7.149 ± 3.381 **E** | -0.343 ± 0.128 **O** |
| visibility | +0.082 ± 0.004 **E** | n/a | +5.299 ± 0.444 **E** | +8.540 ± 1.479 **E** | +0.212 ± 0.018 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.050 ± 0.019 **O** | +0.038 ± 0.008 **E** | -1.020 ± 0.210 **O** | -0.389 ± 0.231 | +0.172 ± 0.028 **E** |
| gradmag | -0.025 ± 0.020 **O** | +0.081 ± 0.017 **E** | +1.503 ± 0.840 **E** | +7.227 ± 1.038 **E** | -0.146 ± 0.083 **O** |
| visibility | +0.120 ± 0.021 **E** | +0.118 ± 0.010 **E** | -0.569 ± 0.220 **O** | +5.657 ± 0.951 **E** | +0.168 ± 0.019 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.068 ± 0.022 **O** | +0.047 ± 0.011 **E** | -0.796 ± 0.178 **O** | -0.372 ± 0.361 | +0.147 ± 0.023 **E** |
| gradmag | -0.017 ± 0.020 | +0.059 ± 0.015 **E** | +0.744 ± 0.609 | +5.004 ± 1.094 **E** | -0.146 ± 0.088 **O** |
| visibility | +0.128 ± 0.027 **E** | +0.098 ± 0.011 **E** | -0.387 ± 0.237 **O** | +5.620 ± 0.916 **E** | +0.140 ± 0.017 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: stated meaning not supported (direction opposite to expected)