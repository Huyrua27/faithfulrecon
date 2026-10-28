# D2 protocol v2.1 — dtu scan37 v8

- 8 training views (farthest-point subset) · base models and signals from runs/dtu_scan37_v8
- Seeds: [0, 1, 2, 3, 4, 5] · K = 5% · removal · snapshots ['0', '200', '1000'] steps · 5 random selections per seed
- Weight mask: selected Gaussians carry ≥ 20% of the pixel's blending weight (footprint mask of Phase 0 reported for comparison)
- Values: mean ± std over seeds; gains = signal − mean of random selections in the same seed.
- Class: **E** = consistent, expected sign (+) · **O** = consistent, opposite sign · · = inconsistent

## Region size and image locality

| signal | region (weight) | region (footprint) | LR_img weight_min @ 1000 (random) | LR_img footprint @ 1000 (random) |
|---|---|---|---|---|
| opacity | 0.00% (random 2.16%) | 32.04% | 0.769 (0.853) | 0.764 (0.744) |
| gradmag | 8.76% (random 2.16%) | 17.62% | 0.811 (0.853) | 0.770 (0.744) |
| visibility | 0.07% (random 2.16%) | 12.43% | 0.908 (0.853) | 0.763 (0.744) |

## Measurability (image SNR_in, weight_min mask)

| signal | step 200 | step 1000 | measurable |
|---|---|---|---|
| opacity | +1.5 ± 0.8 | +1.5 ± 0.5 | no |
| gradmag | +11.8 ± 2.2 | +9.8 ± 1.1 | yes |
| visibility | +6.2 ± 1.6 | +5.1 ± 1.1 | yes |
| random | +12.4 ± 1.9 | +10.5 ± 1.5 | |

## Gains at step 0 (right after removal, no re-optimization)

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | +0.030 ± 0.002 **E** | n/a | +3.305 ± 0.256 **E** | -0.075 ± 0.136 | +0.180 ± 0.024 **E** |
| gradmag | -0.006 ± 0.010 | n/a | -7.724 ± 3.033 **O** | +10.548 ± 7.136 **E** | -0.317 ± 0.143 **O** |
| visibility | +0.045 ± 0.002 **E** | n/a | +3.483 ± 0.273 **E** | +3.884 ± 1.301 **E** | +0.179 ± 0.025 **E** |

Direction gains ×10⁻³.

## Gains at step 200

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.060 ± 0.033 **O** | +0.036 ± 0.016 **E** | -0.427 ± 0.158 **O** | -0.152 ± 1.152 | +0.116 ± 0.022 **E** |
| gradmag | -0.043 ± 0.017 **O** | +0.089 ± 0.024 **E** | +0.640 ± 0.806 | +3.039 ± 4.344 | -0.218 ± 0.131 **O** |
| visibility | +0.055 ± 0.007 **E** | +0.087 ± 0.050 | -0.301 ± 0.113 **O** | +2.721 ± 1.251 **E** | +0.114 ± 0.014 **E** |

Direction gains ×10⁻³.

## Gains at step 1000

| signal | Gain_LR image (region mask) | Gain_LR 3D displacement | Gain_Dir held-out 1−SSIM | Gain_Dir local accuracy | Gain_Dir local completeness |
|---|---|---|---|---|---|
| opacity (n.m.) | -0.084 ± 0.038 **O** | +0.073 ± 0.013 **E** | -0.379 ± 0.083 **O** | +0.069 ± 0.479 | +0.111 ± 0.038 **E** |
| gradmag | -0.042 ± 0.020 **O** | +0.077 ± 0.022 **E** | +0.597 ± 0.606 | +2.278 ± 3.936 | -0.187 ± 0.112 **O** |
| visibility | +0.055 ± 0.012 **E** | +0.105 ± 0.027 **E** | -0.263 ± 0.082 **O** | +3.029 ± 0.915 **E** | +0.102 ± 0.022 **E** |

Direction gains ×10⁻³.

## Decision

**D2: GO** (rule at the top of scripts/analyze_d2.py, primary step 1000).

- **opacity**: not measurable
- **gradmag**: stated meaning not supported (direction opposite to expected)
- **visibility**: stated meaning not supported (direction opposite to expected)