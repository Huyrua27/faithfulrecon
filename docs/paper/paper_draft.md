# FaithfulRecon — working paper draft (Paper A)

> Status: working draft, 28 Oct 2026. Numbers are taken from `results/` (synthetic: `results/d2v21_v8_*`;
> DTU: `results/dtu_multiscan/`). Citations marked † are not yet verified and must be checked against
> `docs/paper/references.md` before submission. Wording rules: no "causal", no "first", no "prove";
> Observation and Interpretation are kept separate.

---

## Abstract

Per-Gaussian reliability signals — scores that rank the primitives of a 3D Gaussian Splatting (3DGS)
model by how much they can be trusted — are increasingly used to prune, densify, or select views.
They are usually validated in one of two ways: by their correlation with an error label, or by the
image quality after acting on them. We argue that neither is sufficient and that the two can
disagree with the geometry itself. We propose an intervention protocol, D2, that removes the
primitives a signal selects, continues optimization from the same checkpoint, and compares the
outcome with size-matched random removals in the same seed. The comparison is made on three axes:
locality (does the change stay where the signal pointed), image error on held-out views, and
reference geometry decomposed into local accuracy and local completeness. We pair D2 with a
correlation-based evaluation, D1, against labels derived from reference geometry.

On eight DTU scans with eight training views (six seeds each), removing Gaussians with large
positional gradients improves held-out image error in 7/8 scans but reduces local completeness in
8/8 scans. Removing rarely observed Gaussians improves both local accuracy and completeness in 8/8
scans, while its effect on held-out images is inconsistent (3/8). Both primary geometric hypotheses
hold for both signals after Holm correction across scans (p = 0.031, n = 8). The correlation-based
view gives a matching picture only when labels are defined by distance to the reference surface. A
commonly used floater label that also requires held-out contribution and opacity reverses the
ranking of signals, and it rates every contribution- or information-based signal below chance by
construction. A depth analysis shows that at least half of the off-surface Gaussians on DTU lie in
front of the true surface: these are depth-ambiguous primitives that fit the sparse training views.
Our results suggest that reliability signals for 3DGS should be judged by intervention and by
decomposed geometry, and that label definitions must be reported as part of the result.

---

## 1. Introduction

3DGS represents a scene as a set of anisotropic Gaussians that are optimized to reproduce the
training images [Kerbl et al. 2023†]. When the views are few or poorly spread, the optimization is
under-constrained. Primitives can explain the training images while sitting at the wrong depth,
floating in free space, or leaving parts of the surface uncovered. Many methods therefore compute a
per-Gaussian score and act on it, for example:

- pruning low-significance Gaussians for compression [LightGaussian, Fan et al. 2024];
- pruning by Fisher-based sensitivity [PUP 3D-GS, Hanson et al. 2025];
- using information or uncertainty to choose views or guide densification [FisherRF, Jiang et al. 2024†].

Such a score carries an implicit claim: the Gaussians it ranks highest are the ones that are
wrong, or the ones whose removal is harmless. That claim is rarely tested directly. Two evaluation
habits dominate:

1. **Correlation with an error label.** For example, the AUROC of the score against
   "floater/not floater". This depends on how the label is defined, and labels are often built
   from the same quantities (opacity, contribution) that the signals use.
2. **Image quality after acting on the score.** For example, PSNR after pruning. This is a
   statement about rendered images, not about the geometry the Gaussians represent. With sparse
   views, the two can come apart.

We study this gap with an intervention protocol borrowed in spirit from ROAR ("remove and retrain")
for feature attributions [Hooker et al. 2019]. We remove the Gaussians a signal selects, continue
optimizing, and compare the result with random removals of the same size. Unlike ROAR, we act on
model primitives rather than input features. We continue from the checkpoint instead of
retraining, we track the identity of every primitive so that we can measure where the change
happens, and we measure the *direction* of the change on decomposed geometry rather than only a
drop in quality.

**Contributions.**
1. **D2, an intervention protocol for per-primitive reliability signals in 3DGS.** It compares
   against size-matched random removals in the same seed, uses a paired no-removal reference and a
   noise floor, and runs a measurability check before any directional claim. It reports locality,
   held-out image error, and local accuracy and completeness separately.
2. **A benchmark with reference geometry.** It covers a procedural synthetic scene (two view
   configurations) and eight DTU scans in an eight-view setting, with per-Gaussian distance labels,
   alignment checks, and a label-definition sweep.
3. **Empirical findings.**
   - Image-based and geometry-based conclusions about the same intervention disagree, in opposite
     directions for two common signals.
   - Correlation-based conclusions depend on the label definition, to the point of reversing the
     ranking of signals.
   - Most off-surface Gaussians in the sparse-view DTU setting are depth-ambiguous primitives in
     front of the surface, which a simple view-count signal detects (mean AUROC 0.88).

---

## 2. Related Work

**Pruning and importance in 3DGS.**
- *LightGaussian* [Fan et al. 2024] assigns each Gaussian a global significance: its hit count over
  the training pixels, weighted by opacity and normalized volume. It prunes low-significance
  Gaussians and then fine-tunes. This score is not Hessian- or Fisher-based.
- *PUP 3D-GS* [Hanson et al. 2025] scores each Gaussian by the log-determinant of a Fisher block
  over its mean and scale, and prunes low-sensitivity Gaussians.
- Both are evaluated by rendering quality and speed after compression. Neither asks whether the
  pruned Gaussians were erroneous, nor what pruning does to the reconstructed geometry. We use
  adaptations of both as baseline signals (Section 4.3).

**Uncertainty and information in radiance fields.**
- *FisherRF*† [Jiang et al. 2024] uses Fisher information of the rendering with respect to the
  parameters to select informative views.
- Other works estimate uncertainty for NeRF/3DGS and evaluate it by correlation with image error
  (AUSE-style curves)†.
- Our D1 belongs to this correlation family. Our point is that its conclusions depend on the label,
  and that it can disagree with intervention results.

**Evaluating explanation and attribution methods.**
- *ROAR* [Hooker et al. 2019] removes the input features an attribution marks as important,
  retrains, and compares with random removal. Many attribution methods were no better than random.
- *Sanity checks for saliency maps*† [Adebayo et al. 2018] showed that visually convincing
  attributions can be insensitive to the model.
- D2 transfers the "intervene and compare with random" principle to model primitives in an
  explicit 3D representation. There, geometry gives an external reference that image-based
  evaluation of attributions lacks.

**Sparse-view reconstruction and floaters.**
- Sparse-view 3DGS and NeRF methods add depth, smoothness, or co-visibility regularizers to reduce
  floaters†.
- We do not propose a new regularizer. We characterize what the off-surface Gaussians are on DTU
  (Section 5.4) and which signals find them. This can inform such methods.

---

## 3. Problem setting

Given a trained 3DGS model with Gaussians $\{g_i\}_{i=1}^N$ and a signal $s:\{g_i\}\to\mathbb{R}$
(higher = less reliable), we ask three questions:

- **RQ1 (prediction).** Does $s$ rank erroneous Gaussians first, under labels derived from reference
  geometry? *(D1)*
- **RQ2 (locality).** When the Gaussians selected by $s$ are removed and the model is re-optimized,
  does the change stay near them, more than for random removals? *(D2)*
- **RQ3 (direction).** Does that change move the model toward the reference geometry: higher local
  accuracy and higher local completeness? Does it improve held-out images? *(D2)*
- **RQ4 (dissociation).** Do RQ1 and RQ2–3 agree? Do image-based and geometry-based judgments
  agree?

**Operational definition.** A signal is *supported* under intervention if its removals are
measurable (above the noise floor in every seed), localized, and move the primary geometric metrics
in the expected direction, consistently across units. This gives evidence under the tested
interventions, not causal identification.

---

## 4. Method

### 4.1 Renderer and training

- **Renderer.** A differentiable 3DGS rasterizer written in pure PyTorch:
  - EWA projection and front-to-back alpha compositing;
  - enumeration of (Gaussian, pixel) pairs with α ≥ 1/255;
  - sorting by (pixel, depth);
  - transmittance as a segmented cumulative sum of log(1−α) in float64.
  - It returns, for every Gaussian and view, its blending contribution Σ_p α·T.
  - Parity with a dense reference: image error 2.4e-7, gradient error < 1e-6.
  - An optional gsplat backend is checked against it.
- **Training.**
  - Standard 3DGS-style optimization with L1 + D-SSIM loss (λ = 0.2), Adam, adaptive
    densification and pruning, and periodic opacity reset.
  - DTU: 7000 iterations, random initialization with 10k points in the normalized unit box, at
    most 300k Gaussians.

### 4.2 D1: correlation with geometric labels

For each Gaussian we compute $d_i$, the distance from its mean to the nearest reference surface
point. Labels:

| Label | Definition | Role |
|---|---|---|
| distance | $d_i > \tau$, $\tau \in \{0.03, 0.1\}$ (normalized units) | primary (from this draft on) |
| $f_i$ ("floater") | $d_i > \tau$ **and** held-out contribution > 0.01 **and** opacity > 0.05 | secondary; reported because it is common, with its bias stated |
| $m_i$ (under-observed) | seen in fewer than k = 2 training views or within θ = 10° | defined; occlusion test pending |

Metrics: AUROC, AUPRC, precision and enrichment at top-k, and Spearman correlation with $d_i$.
Each is compared with random scores in the same model.

### 4.3 Signals evaluated

Convention: higher = less reliable.

| Signal | Definition | Source |
|---|---|---|
| random | uniform | control |
| opacity | $1-\alpha_i$ | heuristic |
| gradmag | mean NDC view-space positional gradient norm over training views where $g_i$ contributes | densification criterion of 3DGS |
| visibility | −(number of training views with contribution > 1e-3), random tie-break | heuristic (observation count) |
| contrib | −Σ_views contribution | heuristic |
| LightGaussian-style | −Σ_views contribution × $\min(V_i/V_{90},1)^{0.1}$ | adaptation of [Fan et al. 2024] |
| gradient inconsistency | trace of the covariance of per-view positional gradients / (‖mean‖² + δ) | Quỳnh Anh |
| fisher (FisherRF-style) | −log(Tr $F_i$ + ε), diagonal over all parameters | adaptation of [Jiang et al. 2024†] (Phát) |
| pup (PUP-style) | −log det($F_i^{\mu,s}$ + εI), 6×6 block over mean and scale | adaptation of [Hanson et al. 2025] (Phát) |
| curvature | Tr$((F_i^{\mu}+\epsilon I)^{-1})$ | Phát |

The Fisher blocks are $F_i = \sum_{\text{views}} J_i^\top J_i$, with $J_i = \partial I/\partial\theta_i$.
They are estimated with Rademacher probes ($g = J^\top v$, so $\mathbb{E}[g_i g_i^\top] = F_i$) using
16 samples per view. At 16 samples, the rank correlation with a 256-sample reference is 0.998 or
higher. These are documented adaptations, not reproductions of the authors' kernels.

### 4.4 D2: intervention protocol (v2.1)

For each seed $s$, signal, and base checkpoint:

1. **Selection.** Take the top K = 5% of Gaussians by the signal. Also draw 5 random selections of
   the same size (generator seeded 10000+s, shared across signals).
2. **Removal and continued optimization.** Prune the selection and continue training from the
   checkpoint:
   - Adam state of the survivors kept;
   - densification and opacity reset disabled;
   - position learning rate at its final value;
   - optimization seed fixed.
   Snapshots are taken at T ∈ {0, 200, 1000} steps (T = 1000 is primary).
3. **References.**
   - *null_a*: continued optimization without removal (seed 1000+s). This is the paired reference
     for every comparison.
   - *null_b*: the same, with seed 2000+s. Its difference from null_a is the noise floor.
4. **Region.** The held-out pixels where the selected Gaussians carry at least 20% of the blending
   weight in the base model. The region is grown to at least 1% of pixels, so that tiny regions do
   not produce unstable ratios.
5. **Measures** (each relative to null_a):
   - **Measurability.** SNR_in = image change inside the region / noise floor. It must be ≥ 2 in
     every seed.
   - **Locality.**
     - LR_img: fraction of the held-out image change that falls inside the region.
     - LR_3D: fraction of Gaussian displacement that falls within radius r = 0.05 of the removed
       Gaussians.
   - **Direction.**
     - held-out 1−SSIM;
     - local accuracy and local completeness: the two Chamfer components between the Gaussian
       means and the reference points, restricted to radius r around the removed Gaussians.
6. **Gains.** Gain = (metric for the signal) − (mean over the 5 random selections in the same seed).
   The sign is oriented so that positive = better than random.
7. **Classification.**
   - Per metric: **E** (all units positive), **O** (all units negative), or inconsistent.
   - Per signal: not measurable, supported, or not supported.
   - **Primary metrics**, fixed before the DTU experiments: local completeness and LR_3D.
   - **Statistics across scans:** two-sided Wilcoxon signed-rank on per-scan mean gains, Holm
     correction over the 2 signals × 2 primary metrics.

### 4.5 Depth-side analysis

To find out where off-surface Gaussians are, we render a z-buffer of the reference points in each
training view (3×3 min-filter to close holes). For each Gaussian we compute the median, over
training views where it projects inside the object mask, of $z_{\text{gauss}} - z_{\text{surface}}$.
With tolerance 0.03, this classifies the Gaussian as in front of, on, or behind the surface.

---

## 5. Experiments

### 5.1 Setup

| | Synthetic | DTU |
|---|---|---|
| Scene | procedural: sphere, box, 3 rods; 40k ground-truth Gaussians | scans 24, 37, 40, 55, 63, 65, 69, 83 (IDR preprocessing) |
| Views | 8 training (az360: full circle; az180: half circle), 16 held-out; 160 px; FOV 30° | 8 training by farthest-point sampling, 16 held-out; 400×300 |
| Reference | analytic surface | official STL, mapped by inverse scale matrix and cropped by masks (alignment: 96.8% of points inside masks, IoU 0.85) |
| Seeds | 9 runs over 2 conditions (D2) | 6 per scan → 48 base models |

### 5.2 D2 on DTU: image and geometry disagree

Mean gain over 8 scans at T = 1000. Count = number of scans with a positive gain.

| Signal | LR_img | LR_3D | held-out 1−SSIM (×10⁻³) | local acc. (×10⁻³) | local comp. |
|---|---|---|---|---|---|
| gradmag | −0.008 (3/8) | **+0.087 (8/8)** | +0.58 (7/8) | +1.87 (6/8) | **−0.214 (0/8)** |
| visibility | +0.098 (8/8) | **+0.090 (8/8)** | +0.02 (3/8) | +5.37 (8/8) | **+0.159 (8/8)** |
| opacity (not measurable) | −0.052 (2/8) | +0.079 (8/8) | −0.45 (0/8) | −0.14 (5/8) | +0.173 (8/8) |

Primary tests (n = 8 scans, two-sided Wilcoxon, Holm over 4 tests):

| Signal | Metric | Direction | p | p_Holm |
|---|---|---|---|---|
| gradmag | local completeness | worse than random, 8/8 | 0.0078 | 0.031 |
| gradmag | LR_3D | better, 8/8 | 0.0078 | 0.031 |
| visibility | local completeness | better, 8/8 | 0.0078 | 0.031 |
| visibility | LR_3D | better, 8/8 | 0.0078 | 0.031 |

**Observation.**
- Removing high-gradient Gaussians damages held-out images immediately in every scan (T = 0:
  −2.6 to −28.8 ×10⁻³). After 1000 steps the images recover beyond random removal (7/8 scans).
  Local completeness stays worse than random in all scans.
- Removing rarely observed Gaussians improves local accuracy and completeness in all scans, while
  held-out image error changes inconsistently.
- Opacity-based selections are not measurable: their weight region is near zero in all scans.

**Interpretation.** An evaluation based on held-out images alone would favor gradmag. Decomposed
geometry shows that gradmag removes surface-carrying Gaussians: re-optimization recovers the
images but not the coverage. Visibility removes Gaussians whose absence the images barely register,
but whose removal brings the geometry closer to the reference.

**Synthetic (exploratory, 9 runs, 2 conditions).** Gradmag shows the same pattern: local
completeness is worse than random in 9/9 runs (az360 −0.239), LR_3D is better (+0.093), and held-out
1−SSIM at T = 0 is −6.8 ×10⁻³. Visibility shows no consistent effect on synthetic data, and opacity is
not measurable. The primary metrics were chosen after the synthetic runs, so the synthetic results
are exploratory. DTU is the confirmatory data.

### 5.3 D1 on DTU: conclusions depend on the label

Mean AUROC over 8 scans (6 seeds each). Random scores give 0.50 ± 0.003.

| Signal | $d>0.1$ | $d>0.03$ | $f_i$ | Spearman with $d$ |
|---|---|---|---|---|
| visibility | **0.878** (0.72–0.97) | 0.815 | 0.658 | +0.60 |
| gradmag | 0.427 (below 0.5 in 8/8) | 0.445 | **0.606** | −0.12 |
| opacity | 0.500 | 0.473 | 0.273 | −0.05 |

Additional signals, scan24 only (6 seeds):

| Signal | $d>0.1$ | $f_i$ |
|---|---|---|
| contrib | 0.652 | 0.410 |
| LightGaussian-style | 0.619 | 0.399 |

Phát's signals on synthetic v8_az360 (3 seeds), $f_i$ only: fisher 0.30, pup 0.31, curvature 0.34.
On DTU, distance labels are pending.

**Observation.**
- Under distance labels, D1 agrees with D2: visibility is good, gradmag is below chance.
- Under $f_i$, gradmag looks useful (0.61), and every contribution- or information-based signal
  falls below 0.5.

**Interpretation.** $f_i$ requires held-out contribution > 0.01 and opacity > 0.05. Gaussians with
little contribution, information, or opacity are therefore excluded from the positive class by
construction. This biases every signal built on those quantities. We use distance labels as
primary, and report $f_i$ only together with this caveat.

### 5.4 What the off-surface Gaussians are (DTU scan24)

- About 51% of Gaussians in scan24 are farther than 0.1 from the reference surface. Across the
  eight scans the fraction ranges from 7.3% (scan83) to 59.9% (scan37).
- Their colors are not background-white (11% near-white), 97% project inside the object masks, and
  their opacity is similar to on-surface Gaussians.
- Depth side, as median over training views:

| Group | in front | on surface | behind | no valid view |
|---|---|---|---|---|
| $d > 0.1$ (≈68k / seed) | **50.0%** | 0.0% | 32.5% | 17.5% |
| $d < 0.01$ (≈41k / seed), control | 0.2% | 86.9% | 12.9% | 0.0% |

**Interpretation.**
- At least half of the off-surface Gaussians sit between the cameras and the surface, carrying
  surface-like color. These are depth-ambiguous primitives that fit the eight training views.
- The control group shows that the "in front" category is clean (0.2% false rate). "Behind" has a
  baseline of about 13% from self-occlusion and z-buffer filtering.
- A plausible reason why visibility detects these primitives is that a wrongly placed Gaussian can
  stay consistent with only a few views. This has not been tested directly yet.

### 5.5 Limitations (current)

- **One view budget (8 views) and one real dataset.** NeRF-Synthetic has no public reference
  geometry. The lego runs did not complete.
- **Confirmatory status.** Scan24 was inspected before the primary hypotheses were written down.
  On the 7 fresh scans alone, the minimum attainable Wilcoxon p is 0.0156, which exceeds the Holm
  threshold. One more fresh scan is needed for a strict confirmatory test.
- **Visibility tie-break.** It is generated on the compute device, and top-5% selections overlap
  only about 82–86% between CPU and GPU. It must be fixed, and visibility re-run.
- **Signal scope.** The literature signals are adaptations. Fisher, PUP, and curvature have not
  been run through D2 yet.
- **Single intervention.** Removal at K = 5% only. A K sweep and other interventions (e.g. opacity
  scaling) are pending.
- **Variation across scans.** Scans 63 and 83 produce about 25k Gaussians, versus 130–175k for the
  other scans, and visibility is weaker there (AUROC 0.72–0.76).

---

## 6. Implementation status

| Component | Location | Owner | Status |
|---|---|---|---|
| PyTorch rasterizer: pair enumeration, log-cumsum transmittance, per-Gaussian contribution; gsplat parity test | `frecon/render/` | Huy | done; 0.99 → 0.19 s/step on GTX 1650 |
| Trainer (densify/prune with Adam-state bookkeeping, opacity reset, checkpoint save/load, step callbacks) | `frecon/train.py` | Huy | done |
| Scenes: synthetic, NeRF-Synthetic loader, DTU loader (RQ decomposition of projection matrices, FPS view split, STL → normalized reference with mask cropping and cache) | `frecon/data/` | Huy | done |
| Baseline signals (opacity, gradmag, visibility) | `frecon/signals/` | Huy | done; visibility tie-break fix pending |
| Gradient inconsistency signal and checkpoint export | `frecon/signals/gradient_inconsistency.py` | Quỳnh Anh | merged (PR #2) |
| Second-order signals (Fisher, PUP-style, curvature, s_geom) with Hutchinson estimator, stability study, export | `frecon/signals/second_order*.py` etc. | Phát | PR #5 under review |
| D1 evaluation (AUROC/AUPRC/precision@k/enrichment, random controls, plots) | `frecon/eval/`, `scripts/run_d1.py` | Tuấn Khải | merged (PR #1) |
| Geometric labels and benchmark | `frecon/benchmark/`, `scripts/make_labels.py` | Trường Thịnh | PR #4 needs fixes (checkpoint fallback, DTU support, label definition) |
| D2 protocol v2.1 (selections, continued optimization, snapshots, region masks, locality/direction metrics, report) | `scripts/d2_v2.py`, `scripts/analyze_d2.py`, `frecon/measure.py` | Huy | done |
| Colab pipeline (Drive-backed runs, multi-process seeds, resumable) | `notebooks/colab_runner.ipynb`, `scripts/run_parallel.py` | Huy | done |
| Multi-scan analysis, depth-side analysis, figures | `results/dtu_multiscan/` | Huy | done (notebook cells; to be moved into `scripts/`) |
| D3 (held-out error prediction) | `frecon/eval/d3.py` | Phát | stub; moved to future work for Paper A |

---

## 7. Plan (28 Oct → mid-December 2026)

**Target.** A complete draft in mid-December, which also serves as the course report. Then
submission to a venue with a later deadline, e.g. ICCV 2027 (deadline to be checked on the official
page).

| Week | Dates | Work | Done when |
|---|---|---|---|
| 1 | 28/10–3/11 | Write down hypotheses for fresh data in `docs/preregistration_dtu.md` and commit before running. Fix the visibility tie-break (CPU jitter) and re-run visibility. Run one more fresh scan (scan97). Switch D1 to distance labels as primary. Merge PR #4 after fixes and PR #5 after clean-up. | 8 fresh scans pass or fail the pre-written tests; labels frozen |
| 2–3 | 4/11–17/11 | Literature signals through D1 **and** D2 on all scans: PUP-style, FisherRF-style, LightGaussian (faithful implementation), gradient inconsistency. Add them to `signals.pt` and set the sample count once (S = 16). Record curvature ties for unseen Gaussians. | Main table with ≥ 7 signals × 8 scans |
| 4 | 18/11–24/11 | Ablations: K ∈ {2%, 5%, 10%}; radius r; one alternative intervention (opacity down-scaling). Utility test: prune with each signal, then measure global accuracy and completeness and held-out PSNR. Test the visibility mechanism (view counts by depth side). | Ablation table; utility figure |
| 5–6 | 25/11–8/12 | Write the paper: method from `method_d2.tex`, figures, related-work verification (all † entries). Move notebook analysis code into `scripts/`. | Full draft |
| 7 | 9/12–15/12 | Advisor feedback, revision, course report derived from the draft. | Report submitted |

**Decision points.**
- If the fresh-scan tests fail for a signal, report the failure; the paper's claim is about
  evaluation, not about a particular signal winning.
- If Fisher or PUP adaptations are unstable on DTU, report them with the stability numbers and
  restrict the claim to the signals that pass the numerical checks.
- If time is short, drop the alternative intervention before dropping the literature signals. The
  literature signals are what reviewers will ask for first.

**Out of scope for Paper A (future work).**
- D3: per-view predictive validity of held-out error.
- Other view budgets and datasets with reference geometry (e.g. Tanks and Temples).
- Using validated signals inside training (densification or view selection).
