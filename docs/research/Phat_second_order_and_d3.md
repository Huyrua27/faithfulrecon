# Second-order signals and D3 — research notes

**Owner:** Phát · **Branches:** `feature/phat-second-order`, `feature/phat-d3`

**Status (27 October 2026):** W3 estimator validated; W4 signals implemented and exported
on all three shared Phase 0 checkpoints, with baseline arrays preserved. The portable
package is verified on Drive and passes the unmodified Tuấn Khải D1 pipeline at `e4ba397`.
Training-scene audits, sample/damping diagnostics, RTX 4060 profiling and handoff receipts
are in `results/second_order/w4/`. GTX 1650 measurement is an explicit user-deferred task;
scientific reviewer acceptance remains open. D3 remains scheduled for W6–W7.
Delivery instructions: `docs/analysis/Phat_W4_handoff_2026-10-27.md`.

Spec: `docs/tasks/Phat_tasks.md`. Code: `frecon/signals/{second_order,fisher,pup_sensitivity,s_geom}.py`,
`frecon/eval/d3.py`, `scripts/run_d3.py`. Tests: `tests/test_second_order.py`, `tests/test_d3.py`.

## Part A — Second-order signals

### A.1 Fisher / Gauss–Newton blocks and the estimator

For training camera `c`, let `I_c` be the flattened rendered RGB image and let
`J_{c,i} = dI_c/dtheta_i` contain the columns for Gaussian `i`. The target is

```text
F_i = sum_c J_{c,i}^T J_{c,i}
```

Each probe `v` has independent Rademacher (+1/-1) entries for every pixel AND channel.
One vector-Jacobian product gives `g_i = J_{c,i}^T v`, so

```text
E[g_i g_i^T] = J_{c,i}^T E[v v^T] J_{c,i} = J_{c,i}^T J_{c,i}
F_hat_i = sum_c (1/S) sum_s g_{c,s,i} g_{c,s,i}^T
```

We average probes **within** each view and **sum** views. Gradients of the actual
L1/SSIM photometric training loss would estimate a different quantity. These are
rendered-image Gauss–Newton blocks; identifying them as Fisher information assumes
an appropriate independent Gaussian observation-noise model and normalization.
They are not the exact Hessian of the training objective. Inter-Gaussian blocks are
omitted, so inverses of these small blocks are not full-scene marginal covariances.

Implementation: `frecon/signals/second_order.py::fisher_blocks`. It accepts unique
names from `model.params` in the requested order, including scalar `(N,)` fields:

| Parameters | Output | Basis |
|---|---|---|
| `("means",)` | `(N,3,3)` | World-space position |
| `("means","log_scales")` | `(N,6,6)` | Position and stored log-scale, including cross terms |

There is no damping, inversion, or reliability-score transformation in the estimator.
If an invisible Gaussian has no image influence, its block is zero. Empty camera
lists, invalid sample counts/names, duplicate names, and frozen selected parameters
are rejected. Accumulation is float32 (float64 is retained for float64 models).

**Scale convention:** `GaussianModel` renders `s = exp(l)`, where `l = log_scales`.
The 6x6 estimator therefore differentiates with respect to `(mu,l)`. To obtain a
block in `(mu,s)`, apply `A^T F_(mu,l) A`, with
`A = diag(1,1,1,1/s_x,1/s_y,1/s_z)`, before choosing damping. The W4 PUP-style signal
must explicitly choose this basis instead of silently treating log-scale as scale.

Caching uses `("fisher", params, S, ctx.seed)`. A context represents an immutable
model/camera/backend configuration: create a new context after any such change, and
do not modify a returned cached matrix. Stable hashed seeds create a dedicated
generator per camera, independent of `ctx.generator` and global RNG consumption.
Different sample budgets reuse probe prefixes on the same device/backend. CPU and
CUDA random streams are not claimed to be bit-identical.

`torch.autograd.grad` obtains the vector-Jacobian products without changing existing
parameter gradients. Parameters, their order/count, and values are preserved. One
render graph is reused for S backward passes, then released before the next view.
Time scales as approximately V renders + V*S backward passes; persistent estimator
storage is O(N*D^2), in addition to one view's renderer graph. Production pixel
Jacobians and per-sample gradients are never retained.

#### Validation on 17 October 2026

The reference independently uses `torch.autograd.functional.jacobian` on a functional
rendering closure and computes exact per-Gaussian `J_i^T J_i`. Fixture: 20 Gaussians,
two 16x16 RGB cameras, scene seed 123. Cameras have no GT image; the estimator does
not require one. Probe seeds are 0, 1, 2, 3; the implementation itself is called for
every estimate, rather than sampling an already-computed reference Jacobian.

Relative Frobenius error `||F_hat-F||_F / ||F||_F`, mean +/- sample std over probe seeds:

| S per view | Position 3x3 | Position/log-scale 6x6 |
|---|---|---|
| 4 | 0.84762 +/- 0.46468 | 0.85486 +/- 0.46078 |
| 64 | 0.17449 +/- 0.03652 | 0.17694 +/- 0.03684 |
| 1024 | 0.04084 +/- 0.00562 | 0.04145 +/- 0.00553 |

The aggregate trend is compatible with Monte Carlo `1/sqrt(S)` convergence. This
single-scene study does not establish a universal rate or adequate production
rankings at the default S=4; S must be evaluated in W4. Tests also verify view/sample
normalization analytically, symmetry/PSD, parameter order, scalar/unused parameters,
invisible primitives, deterministic caching/RNG isolation, and preservation of
model/gradient state. The CUDA test covers a real-renderer smoke check and an
analytical normalization comparison.

Results: `results/second_order/{report.md,convergence.csv,convergence.png,environment.json}`.
The experiment report records the actual environment and limitations. Targeted tests:
19 passed, 6 W4 skips. Full suite: 26 passed, 19 skips across unfinished tracks.

Reproduce in the project environment:

```powershell
python -m pytest tests/test_second_order.py -q
python -m tests.test_second_order --out_dir results/second_order
```

The verified interpreter is `C:/Users/phatl/anaconda3/python.exe` (Python 3.12.7,
PyTorch 2.8.0+cu129). The convergence table is a CPU, one-thread result; CUDA smoke
validation passed separately. Full-scene/GTX 1650 runtime and memory remain W4 work.

### A.2 FisherRF-style signal

`fisher = -log(sum_d F_hat[i,d,d] + 1e-12)` uses all 14 stored parameters:
position (3), log-scale (3), quaternion (4), opacity logit (1), color logits (3).
The diagonal estimator stores `(N,14)` instead of full `(N,14,14)` blocks. It uses
the same image probes and view/sample normalization as A.1. Low information produces
a high unreliability score. Invisible primitives receive `-log(eps)`, not NaN.

[FisherRF](https://arxiv.org/html/2311.17874v2) motivates information from rendered-image
derivatives, diagonal approximations, and regularization for uncertainty/view selection.
The [authors' uncertainty renderer](https://raw.githubusercontent.com/JiangWenPL/FisherRF/main/render_uncertainty.py)
selects DC/SH color fields by default. Our all-parameter trace, SH-degree-0 logits,
training-only views, and stochastic PyTorch derivatives define an **adapted baseline**,
not a reproduction of FisherRF's uncertainty maps or expected information gain.
The trace mixes units and changes under reparameterization; comparisons must retain
the stated basis, view count, resolution, renderer, and epsilon.

### A.3 PUP 3D-GS-style sensitivity

`pup = -sum_k log(lambda_k(F_i) + eps)`, with a 6x6 position/scale block and default
`eps=1e-8`. The default basis is activated scale: the native position/log-scale block
is transformed by the chain rule in A.1 **before** damping. `scale_basis=log_scale`
is an explicit alternative, not an equivalent definition. Spectra are computed on
symmetrized float64 matrices; tiny negative roundoff is clipped to zero before damping.
CUDA matrices are processed in batches of 512 to bound eigensolver workspace.

[PUP 3D-GS](https://arxiv.org/html/2406.10219v3) uses spatial sensitivity and a block
log-determinant to retain important Gaussians during pruning. We negate the sensitivity
to satisfy high = less reliable. Our per-pixel Rademacher approximation, absolute
damping, and PyTorch renderer differ from the paper's patch approximation and custom
implementation. PUP pruning performance does not establish our score's failure
association or intervention response. Invisible blocks give `-6 log(eps)`.

### A.4 Curvature u_i and s_geom

`u_i = sum_k 1/(lambda_k(F_i^mu)+eps)`, with world-space position blocks. The default
`eps=1e-8` gives `u_i=3e8` for a zero block. High values indicate weak local image
constraint in this basis; this is not a full-scene marginal covariance.

The local temporary `v_i` is the population trace variance of position gradients of
the training photometric loss, divided by squared mean gradient plus `delta=1e-12`.
Only views with contribution above `1e-3` enter; the loss uses DSSIM weight 0.2.
With fewer than `n_min=2` eligible views, `v_i=1` denotes unknown evidence. This
convention is finite, but is not a calibrated prior. Counts/providers are exported.
The three checkpoints have 666 / 800 / 638 such Gaussians. Quỳnh Anh's implementation
is used when available; only `NotImplementedError` activates this local fallback,
and her source file is unchanged. Gradient buffers/model values are preserved.

The exported variants are `curvature=u`, `gradient_inconsistency=v`, `s_geom=u*v`,
and `s_geom_ranksum=rank(u)+rank(v)`. Ranks are average ranks for ties, normalized
to [0,1], giving a sum in [0,2]. A constant array receives rank 0.5.
Removing `exp(-lambda*u*v)` preserves per-Gaussian rankings after accounting for
orientation; it need not preserve pixel rankings after alpha blending. D3 must
fix the map normalization/transform explicitly.

### A.5 Export, experiments, and current interpretation

`scripts/export_second_order.py` supports S/epsilon/basis options and validates the
checkpoint SHA256 and Gaussian count before scene loading. `--checkpoint` relocates
the file without weakening these guards or rewriting the original manifest.
`save_scores` atomically merges finite float32 `(N,)` arrays after matching scene,
condition, seed, hash and count; replacement requires `--overwrite`. Existing
baseline arrays and metadata are retained. The standard registry calls the five
implemented second-order names with S=4 defaults; the handoff exporter uses S=16
and records every option. The temporary `v` is provided through Phát's exporter.

The shared package passed transfer checksums (39 files), checkpoint hashes, exact
label-coordinate row alignment, and exact baseline comparison against `signals.pt`.
The committed Phase 0 configuration reproduces held-out PSNR within 0.000046 dB
and recomputes the supplied distances/floater labels. Six new scores plus four
baselines are saved in each ignored `runs/d1/synthetic/v8_az360/seed_{0,1,2}/scores.npz`.
Reports, options, source hashes, and reproduction commands are in
`results/second_order/w4/report.md`.

Initial AUROC against **provisional** `f_i` is below 0.5 for the information/curvature
scores and the product; the local `v` is above 0.5. This is an association diagnostic,
not grounds to flip orientation after observing labels or to claim faithfulness.
Unobserved Gaussians score uncertain but fail the visibility gate in `f_i`, which
is a plausible explanation to test with predeclared eligibility strata. S/epsilon
stability is measured independently of these labels. The current S=16 export is
a documented working setting, not a universally optimal budget. W5 should compare
u/v/product/rank-sum, activated/log-scale basis, and visibility strata before D2.

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
| FisherRF (ECCV'24) | Diagonal image-derivative information | Uncertainty/view acquisition | Reconstruction / acquisition performance | Here: all-parameter trace, per-Gaussian association and intervention evaluation |
| PUP 3D-GS (CVPR'25) | Spatial block log-determinant | Gaussian pruning | Compression and rendering quality | Here: stochastic spatial blocks, reversed score orientation and D1–D3 evaluation |
| Manifold Sampling (SIGGRAPH Asia'24) | | | | |
| Bayes' Rays (CVPR'24) | | | | |
| CF-NeRF | | | | |

## References

- FisherRF: https://arxiv.org/abs/2311.17874; source comparison linked in A.2.
- PUP 3D-GS: https://arxiv.org/abs/2406.10219.
- Other uncertainty methods in Part C remain a later related-work reading task.
