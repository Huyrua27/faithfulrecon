# Second-order signals and D3 — research notes

**Owner:** Phát · **Branches:** `feature/phat-second-order`, `feature/phat-d3` · **Status:** TODO

Spec: `docs/tasks/Phat_tasks.md`. Code: `frecon/signals/{second_order,fisher,pup_sensitivity,s_geom}.py`,
`frecon/eval/d3.py`, `scripts/run_d3.py`. Tests: `tests/test_second_order.py`, `tests/test_d3.py`.

## Part A — Second-order signals

### A.1 Fisher / Gauss–Newton blocks and the estimator
TODO — derivation of E[g g^T] = J^T J; variance vs n_samples (test result); cost.

### A.2 FisherRF-style signal
TODO — definition in the paper · our definition · differences · orientation.

### A.3 PUP 3D-GS-style sensitivity
TODO — same structure.

### A.4 Curvature u_i and s_geom
TODO — definition · hypothesis (not assumption) · variants · ablation results.

## Part B — D3 held-out predictive validity

### B.1 Per-Gaussian held-out error e_i
TODO — attribution trick, choice of err (L1 / 1−SSIM), handling of Gaussians unseen in held-out views.

### B.2 Pixel-level score maps and AUSE
TODO — normalization, blending of scores, oracle curve.

### B.3 Results
TODO — `docs/templates/experiment_report.md`; Observation ≠ Interpretation.

### B.4 D1–D2–D3 table per signal (RQ4)
TODO

## Part C — Related work notes

| Paper | Signal | Used for | Evaluated by | Difference from FaithfulRecon |
|---|---|---|---|---|
| FisherRF (ECCV'24) | | | | |
| PUP 3D-GS (CVPR'25) | | | | |
| Manifold Sampling (SIGGRAPH Asia'24) | | | | |
| Bayes' Rays (CVPR'24) | | | | |
| CF-NeRF | | | | |

## References
TODO
