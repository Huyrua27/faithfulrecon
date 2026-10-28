# Signal audit — Quỳnh Anh

Scene: synthetic / v8_az360; supplied checkpoints, seeds [0, 1, 2].
Recomputed scores use CPU / torch renderer and results/phase0/config_used.yaml.
Checkpoint hashes, Gaussian counts, score types and finiteness are checked by the exporter.

## Checkpoint and evidence counts

| Seed | N | No training-view contribution | Fewer than 2 contributing views |
|---|---:|---:|---:|
| 0 | 27733 | 435 | 666 |
| 1 | 27984 | 488 | 800 |
| 2 | 28691 | 395 | 638 |

## Reproduction of supplied scores

Overlap is intersection / K between supplied and recomputed Top 5% in the SAME checkpoint.
Exact score ties in the exporter use Gaussian ID. This does not replay Phase 0 torch.topk ties.

| Seed | Signal | Max absolute difference | Spearman | Top 5% overlap | Count mismatches |
|---|---|---:|---:|---:|---:|
| 0 | random | 0.996 | 0.004316 | 4.542% |  |
| 0 | opacity | 1.19e-07 | 1.000000 | 100.000% |  |
| 0 | gradmag | 5.81e-05 | 0.999990 | 99.928% |  |
| 0 | visibility | 0.000993 | 0.961266 | 83.706% | 0 |
| 1 | random | 0.996 | 0.007707 | 5.147% |  |
| 1 | opacity | 1.19e-07 | 1.000000 | 100.000% |  |
| 1 | gradmag | 0.000169 | 0.999982 | 99.857% |  |
| 1 | visibility | 0.000994 | 0.959309 | 86.133% | 0 |
| 2 | random | 0.997 | -0.001117 | 4.948% |  |
| 2 | opacity | 1.19e-07 | 1.000000 | 100.000% |  |
| 2 | gradmag | 4.66e-05 | 0.999985 | 99.930% |  |
| 2 | visibility | 1 | 0.961821 | 81.812% | 1 |

### Interpretation of reproduction differences

Opacity is reproduced within float32 activation precision. GradMag rankings and Top 5% are nearly identical, but some gradients differ appreciably in value: this is a ranking reproduction, not bitwise score reproduction.
Random-score differences are consistent with different device RNG streams (the saved config requests CUDA; the original runtime/version is not recorded). The source runtime has not been verified; no exact RNG reproduction is claimed.
Visibility differences are predominantly within tied integer counts; the separate count-mismatch column flags threshold-sensitive numerical differences. Device/renderer arithmetic and threshold sensitivity remain audit items.

## Exploratory association with provisional f_i

Mean ± sample std across seeds. These are sanity checks with supplied provisional labels, not final D1 results.
Random expected precision equals prevalence; the exported random vector is one draw, not the matched D2 control.

| Signal | AUROC | Precision@5% | Enrichment@5% | Zero-visibility fraction in Top 5% |
|---|---:|---:|---:|---:|
| random | 0.4981 ± 0.0071 | 0.0837 ± 0.0083 | 0.9658 ± 0.1083 | 0.0156 ± 0.0012 |
| opacity | 0.3486 ± 0.0204 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2294 ± 0.0157 |
| gradmag | 0.6536 ± 0.0105 | 0.1079 ± 0.0085 | 1.2458 ± 0.1410 | 0.0000 ± 0.0000 |
| visibility | 0.3535 ± 0.0116 | 0.0260 ± 0.0317 | 0.2913 ± 0.3472 | 0.3126 ± 0.0368 |
| gradient_inconsistency | 0.6508 ± 0.0250 | 0.0991 ± 0.0064 | 1.1419 ± 0.0669 | 0.0000 ± 0.0000 |

Gradient inconsistency shows association with provisional floater labels, but no D2 intervention for this new signal has been run. Association does not establish removal benefit or faithfulness.
Opacity's zero Top-5% floater precision must be interpreted with the label gate opacity > .05.
The count<2 fallback is zero evidence of inconsistency, not evidence of reliable geometry.

## Gradient inconsistency: initial confounder checks

| Seed | Spearman vs opacity score (1-alpha) | vs visibility count | vs GradMag | Top 5% mean-gradient squared < delta |
|---|---:|---:|---:|---:|
| 0 | -0.234 | 0.400 | 0.279 | 3.53% |
| 1 | -0.199 | 0.408 | 0.267 | 3.79% |
| 2 | -0.237 | 0.404 | 0.281 | 2.65% |

The moderate positive association with visibility count deserves a stratified ablation. Marginal correlations do not establish domination or independence. Near-zero mean gradients occur in a small portion of Top 5%; delta sensitivity still requires an explicit sweep.

## Phase 0: original interventions, original selections

These values are read from supplied metrics.json; interventions were NOT rerun using CPU selections.
Measurable requires image SNR_in >= 2 in every seed. Gains below that gate are not interpreted.

| Signal | Seed | SNR_in | Gain LR (3D) | Gain Dir (local CD) | Measurable in this seed |
|---|---:|---:|---:|---:|---|
| opacity | 0 | 1.164 | +0.028996 | +0.00002025 | no |
| gradmag | 0 | 4.977 | +0.072760 | -0.00008883 | yes |
| visibility | 0 | 2.259 | +0.035412 | +0.00042901 | yes |
| opacity | 1 | 1.056 | +0.061550 | +0.00009597 | no |
| gradmag | 1 | 4.272 | +0.104209 | -0.00021718 | yes |
| visibility | 1 | 1.917 | -0.052136 | +0.00159696 | no |
| opacity | 2 | 1.091 | +0.047378 | +0.00003532 | no |
| gradmag | 2 | 4.879 | +0.105862 | -0.00008806 | yes |
| visibility | 2 | 1.143 | +0.023148 | +0.00004691 | no |

### Observation

GradMag is measurable in all seeds; 3D locality gains are positive and local-Chamfer direction gains negative. Opacity is near the noise floor. Visibility does not pass the measurability gate in all seeds.

### Interpretation

GradMag selects influential primitives with some floater enrichment, but their removal harms local geometry relative to matched random removal in this pilot. A possible explanation is mixed selection of erroneous and necessary structural Gaussians; this explanation needs structural inspection, not only AUROC.
Image LR is high for random removal as well; high LR alone does not show discriminating locality. Separate 3D locality and mask coverage are required. Three seeds support pilot observations only.

### Limitations and next actions

- The supplied m_i is unavailable and all False; no m_i evaluation is performed.
- Original execution environment and config provenance are incomplete; the committed saved config is an assumption.
- Do not compare Gaussian IDs across independently trained seeds. Compare metrics/distributions across seeds.
- Keep existing D2 unchanged. Propose immediate-removal measurement and narrower masks to the D2 owner.
- Test visibility tie seeds, contribution threshold, and GI delta in separate ablations.
- Test whether GI ranking is dominated by visibility, opacity or near-zero mean gradient before interpretation.
