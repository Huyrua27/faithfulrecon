# Verified references and related-work notes

Single source of truth for citations. **Only cite entries marked "verified".** Add the source used
to verify each entry. Descriptions are written from the papers (or their abstracts), not from
secondary summaries.

## Verified

| Key | Citation | Verified via | What it does | Relation to FaithfulRecon |
|---|---|---|---|---|
| `hooker2019roar` | S. Hooker, D. Erhan, P.-J. Kindermans, B. Kim. *A Benchmark for Interpretability Methods in Deep Neural Networks.* NeurIPS 2019. arXiv:1806.10758 (earlier title: *Evaluating Feature Importance Estimates*) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html), [arXiv](https://arxiv.org/pdf/1806.10758v2) | ROAR (RemOve And Retrain): remove the input features an attribution method marks as important, **retrain**, and measure the accuracy drop; compared against a random assignment of importance. Many attribution methods were no better than random. | Closest methodological precedent for D2: intervene on what a signal selects and compare with random selection. Differences: D2 acts on model *primitives* (not input features), **continues optimization from the checkpoint** instead of retraining from scratch, keeps primitive identities to measure locality, and measures the *direction* of the change (accuracy vs completeness), not only a drop. |
| `fan2024lightgaussian` | Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang. *LightGaussian: Unbounded 3D Gaussian Compression with 15× Reduction and 200+ FPS.* NeurIPS 2024. arXiv:2311.17245 | [DBLP](https://dblp.org/pid/321/5079), [arXiv](https://arxiv.org/pdf/2311.17245) | Compression. Each Gaussian gets a **global significance** score: its hit count over all training pixels, weighted by opacity and by its normalized volume (`V = 4/3 π abc` from the scales). Low-significance Gaussians are pruned, followed by a recovery (fine-tuning) step. **Not** Hessian/Fisher-based. | Another importance score used to prune; candidate baseline signal. Evaluated by rendering quality after compression, not by whether pruned Gaussians are erroneous. Check author order on the NeurIPS proceedings page before citing (sources differ). |
| `hanson2025pup` | A. Hanson et al. *PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting.* CVPR 2025. arXiv:2406.10219 | [arXiv](https://arxiv.org/pdf/2406.10219) | Second-order (Fisher) sensitivity of the reconstruction error w.r.t. spatial parameters of each Gaussian; prunes low-sensitivity Gaussians. Reports gains over LightGaussian. | Baseline signal (Phát: `frecon/signals/pup_sensitivity.py`). |

## To verify before citing

| Mentioned as | Status | Note |
|---|---|---|
| "RAGP" (adaptive recovery after pruning) | **Not found** (search 13/10/2026) | No paper with this name. Closest: *Prune Wisely, Reconstruct Sharply* (Wang et al., Univ. Bristol, arXiv:2602.24136) — reconstruction-aware pruning scheduler + 3D Difference-of-Gaussian primitives for recovery. **Do not cite "RAGP".** If needed, read 2602.24136 and add it here. |
| FisherRF (Jiang et al., ECCV 2024) | in overview, not re-checked here | Phát to verify while implementing `fisher.py`. |
| LPM (Yang et al., CVPR 2025) | in overview, not re-checked here | |
| Manifold Sampling (Lyu et al., SIGGRAPH Asia 2024) | in overview, not re-checked here | |
| Adebayo et al., Sanity Checks for Saliency Maps (NeurIPS 2018) | in overview, not re-checked here | |
| Koh & Liang, Influence Functions (ICML 2017) | in overview, not re-checked here | |
| Guo et al., On Calibration of Modern Neural Networks (ICML 2017) | in overview, not re-checked here | |

## Wording to avoid in related work

- Do not describe D2 as "causal" / "do-operator" / "causal faithfulness". It provides intervention
  evidence under the tested interventions, not causal identification (see `method_d2.tex`).
- Do not say LightGaussian uses Hessian/Fisher information.
