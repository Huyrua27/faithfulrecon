# W4 handoff — Phát, 27 October 2026

**Later PR update:** main now includes the official v and D1 implementations. This
document records the original temporary-v handoff; use
`docs/analysis/Phat_W4_PR_review_2026-10-27.md` for the current official-v package
and validation against main `02a1536`. The original package is preserved unchanged.

The W4 implementation and portable three-seed data package are ready for review.
This closes the delivery work started on 23 October. GTX 1650 profiling remains
an open task, explicitly deferred by the user on 27 October; reviewer acceptance
of the signal definitions is not assumed. D3 remains a separate milestone.

## Delivered artifacts

- Code branch: `feature/phat-second-order`, based on refreshed `origin/main` at `3aef207`.
- [Shared handoff folder](https://drive.google.com/drive/folders/10Rz779ch6FJecUD9kKCFrWpAdgPoJjew).
- [Portable ZIP](https://drive.google.com/file/d/1_8l1ODywKgpzrom4bgB-PLCAEKhKmJTc/view?usp=drivesdk):
  17,448,948 bytes, SHA256
  `600c6311fa659a4aeb1a1b3b0d4df2b5769cf05fb9021209925cb2de35e71a28`.
- Receipt and complete file inventory: `results/second_order/w4/handoff/`.
- Method and paper adaptations: `docs/research/Phat_second_order_and_d3.md`, A.1–A.5.
- Numerical evidence and limitations: `results/second_order/w4/report.md`.

The new Drive folder inherits the existing team's access. Existing data and sharing
were preserved. The ZIP was downloaded for a remote comparison after upload:
one matching file, zero differences. Every archive member also passed SHA256/file-set
verification. There are 17 payload files plus the manifest.

## Requirements and evidence

| W4 requirement | Evidence / acceptance status |
|---|---|
| Shared 3x3 and 6x6 image-Jacobian estimator | Active exact-Jacobian, convergence, normalization, RNG/cache and state-preservation checks; W3 report |
| Fisher, PUP, curvature, s_geom contract | All five registered second-order variants run through `compute_signals`; six handoff arrays including standalone temporary v are finite float32 `(N,)` |
| Explicit parameter basis and paper differences | All-parameter stored-basis Fisher trace; activated-scale PUP with chain rule; stochastic adaptation and damping documented |
| u/v/product/rank-sum and S/epsilon sensitivity | Three probe seeds at S=4/16/64, eps=1e-8/1e-6/1e-4 on constructed fixture and trained checkpoint 0; approximate S=256 reference |
| D1 ingestion without consumer code edits | Unmodified Tuấn Khải branch at `e4ba39771944a6eb084f29ee3eb87a46b984302f`: exit 0, 30 rows, all ten signals and all three seeds |
| Checkpoint identity and row order | Original SHA256/N and exact label xyz alignment verified; every packaged checkpoint is included and verified by the consumer |
| Preserve baseline scores | All four baselines byte-equivalent as arrays to the supplied source; consumer AUROCs match prior diagnostics exactly |
| Portable delivery | Relative `base.pt` manifest paths, original source manifests retained separately, config and SHA256 inventory included; ZIP verified remotely |
| Research + code + tests + experimental output + reproduction | Included in branch; full suite **54 passed, 14 skipped**, including five portable-package regressions |
| GTX 1650 runtime/memory | **Open, user-deferred**; current measured hardware is RTX 4060 Laptop |
| Scientific review / official v integration | **Open review item**; temporary v is allowed by the assignment and its provider/options are explicit |

The consumer test uses an **unmerged evaluation branch**. Main's `run_d1.py` is
still a stub. This is an interoperability check against the teammate's actual
implementation, not a claim that D1 now runs on main or that the other track is complete.

## Package contents and use

```text
README.md
manifest.json
config.yaml
d1/synthetic/v8_az360/seed_{0,1,2}/
  base.pt
  checkpoint.json          # path="base.pt", original SHA256/N
  checkpoint.source.json   # original source manifest, unchanged
  scores.npz               # 4 baselines + 6 new arrays + metadata
  geometric_labels.npz
```

The package includes the three base checkpoints so D1 verifies actual bytes after
extraction without editing paths. Packaging never rewrites the source runs.
`scores.npz` and labels are copied unchanged. Large files remain outside Git.

From the feature branch, verify/downloaded ZIP and extract into a new directory:

```powershell
python scripts/package_second_order_handoff.py --verify runs/handoff/w4-second-order-2026-10-27.zip
python -m zipfile -e runs/handoff/w4-second-order-2026-10-27.zip runs/handoff/received-w4
```

Reproduce the consumer check without editing or merging another track's files.
The snapshot directory is hidden to avoid collecting its tests in this branch:

```powershell
git fetch --no-tags origin
git archive --format=zip --output=runs/handoff/d1-consumer-source.zip e4ba39771944a6eb084f29ee3eb87a46b984302f
python -m zipfile -e runs/handoff/d1-consumer-source.zip runs/handoff/.d1-consumer-source
python runs/handoff/.d1-consumer-source/scripts/run_d1.py --scene synthetic --condition v8_az360 --runs_root runs/handoff/received-w4/d1 --out_dir runs/handoff/d1-consumer-results --n_random 100
```

The completed check emitted exactly the expected provisional-label and unavailable-m_i
warnings. It emitted no missing-checkpoint or unverified-hash warning. Inputs were
unchanged. Full provenance and the 30 aggregate rows are in
`results/second_order/w4/handoff/{consumer_validation.json,consumer_metrics.csv}`.

To rebuild from the original shared source and existing W4 exports:

```powershell
python scripts/package_second_order_handoff.py --out_dir runs/handoff/new-w4-package
```

The builder rejects missing/invalid scores, changed baselines, mixed identities,
reordered label coordinates, unavailable-but-nonempty m_i, configuration mismatch
and existing output paths. The verifier checks exact archive membership and hashes.

## Signal conventions and limitations for review

Handoff settings: S=16, PUP activated-scale basis, PUP/curvature eps=1e-8,
Fisher eps=1e-12, v delta=1e-12 and n_min=2. The temporary v uses population gradient
variance over training views contributing above 1e-3; fewer than two views gives
v=1 (unknown). Provider `temporary_phat` and counts 666 / 800 / 638 are recorded.
Quỳnh Anh's file and exporter remain unchanged.

Labels are provisional, with `m_i_available=False`; D1 must skip m_i metrics.
Numerical correctness and rank stability do not imply failure prediction. Fisher,
PUP, curvature and the product have AUROC below 0.5 in the supplied provisional
diagnostic. Score orientation was kept as specified. D2 response and D3 held-out
prediction have not been established by this handoff.

## Remaining tasks

1. **GTX 1650 measurement (user-deferred):** run the trained three-seed benchmark on
   that hardware. Record exact GPU/software, synchronized runtime and PyTorch memory
   peaks, using S=16 and the same data/config. Do not relabel RTX 4060 results.

   ```powershell
   nvidia-smi --query-gpu=name,memory.total --format=csv
   python scripts/validate_second_order_checkpoints.py --source_root runs/drive_source/data --out_root runs/gtx1650/d1 --results results/second_order/w4/gtx1650
   ```

2. **Huy / Quỳnh Anh review:** confirm the temporary v unknown-view convention and
   parameterization before using it as the agreed cross-track definition. Future
   official-v exports must retain provider/version metadata.
3. **W5 follow-up:** visibility-stratified u/v/product/rank-sum and scale-basis
   ablations, then frozen settings for D2. D3 remains due in W6–W7.

No teammate notification or reviewer approval is claimed by these delivery checks.
