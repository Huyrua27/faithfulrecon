# Quỳnh Anh — signal outputs

Start with [research definitions and reproduce commands](../../docs/research/QuynhAnh_signal_research.md), [experiment audit](QuynhAnh_signal_audit.md), [validation](validation.md), and [implementation notes](../../docs/notes/QuynhAnh_notes.md).

## Files

- `checkpoint_audit.csv`: checkpoint identities and contributing-view counts.
- `reproduction_comparison.csv`: supplied vs recomputed score/ranking comparison.
- `signal_sanity.csv`: exploratory f_i association, Top1/5/10 precision and enrichment; provisional labels only.
- `confounder_checks.csv`: GI vs opacity, visibility and GradMag; denominator diagnostics.
- `phase0_per_seed.csv`: original D2 results, paired with original random removals.
- `recomputed/`: four representative seed-0 / Top-5% scatter figures: opacity, GradMag, visibility and GI.
- `cached/`: four seed-0 / Top-5% figures from the original supplied scores, including random.
- `figure_manifest.csv`: paths, sizes, SHA256 and inclusion status for all 81 experiment figures.

## Full figure archive (Google Drive)

**Drive folder:** [faithfulrecon — complete signal figures](https://drive.google.com/drive/folders/1PNzTQDJtbtzddxTgP39xudricTks5p5E?usp=sharing).

**Upload status: pending.** The supplied folder was opened successfully, but the browser session is not signed in and has no upload control. The ZIP is prepared locally; it has not been uploaded by the agent. Upload it into this folder using a signed-in account with write access, then confirm that the ZIP is visible to reviewers and update this status.

- Local ZIP (gitignored): `runs/signals/archives/QuynhAnh_signal_figures_full.zip`.
- Contents: all **81 PNGs** (36 cached + 45 recomputed), plus a SHA256 manifest.
- Archive size: **16135921 bytes**; SHA256: `34f33c05709eea3db2c0bcc33f93d3e12e76be2db828a7ad36358829b297b0e9`.
- Full extracted figures (gitignored): `runs/signals/figures/{cached,recomputed}/`.
- Figures kept in the PR: **8 PNGs**, **1527940 bytes total** (approximately 1.53 MB).

Upload the ZIP to the Drive folder above and check the intended reviewers can download it before merge. The remaining 73 figures are preserved locally and in the ZIP, rather than deleted from the experiment outputs.

Reproduce the complete image set using `scripts/reproduce_quynhanh.ps1`. It writes all figures under `runs/` and copies only the eight representative figures into `results/signals/`. The default exporter figure directory is also under `runs/`; `.gitignore` permits only the eight representative signal PNGs in `results/`.

Large per-Gaussian outputs are gitignored under `runs/signals/`:

| Seed | Recomputed CSV | Cached CSV |
|---|---|---|
| 0 | [signal_results.csv](../../runs/signals/recomputed/synthetic/v8_az360/seed_0/signal_results.csv) | [signal_results.csv](../../runs/signals/cached/synthetic/v8_az360/seed_0/signal_results.csv) |
| 1 | [signal_results.csv](../../runs/signals/recomputed/synthetic/v8_az360/seed_1/signal_results.csv) | [signal_results.csv](../../runs/signals/cached/synthetic/v8_az360/seed_1/signal_results.csv) |
| 2 | [signal_results.csv](../../runs/signals/recomputed/synthetic/v8_az360/seed_2/signal_results.csv) | [signal_results.csv](../../runs/signals/cached/synthetic/v8_az360/seed_2/signal_results.csv) |

Each run also contains scores.npz, rankings.npz, topk.npz, diagnostics.npz, signal_summary.json and a local checkpoint.json. The per-Gaussian score arrays retain checkpoint row order; ranking arrays contain sorted Gaussian IDs.

## Example: GI Top 5%, seed 0

![GI Top5](recomputed/synthetic_v8_az360_seed0_gradient_inconsistency_top5.png)

Plots show Gaussian centers in world coordinates, with fixed bounds/camera within each seed; they do not modify opacity or measure rendering influence.
