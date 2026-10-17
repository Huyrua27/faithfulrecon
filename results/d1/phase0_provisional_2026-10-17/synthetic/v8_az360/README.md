# Phase 0 provisional D1 evaluation — 17/10/2026

These outputs were computed from the Phase 0 paired exports supplied by the project maintainer for D1 evaluation, for scene `synthetic`, condition `v8_az360`, seeds 0/1/2. They are not the handcrafted QA fixture outputs.

Labels are **provisional**: f_i comes from Phase 0 floater_labels; m_i is unavailable and has no metric rows or figures. The four signals are random, opacity, gradmag and visibility. Matched random uses R=100 draws per seed/label. This is a pipeline check on received experiment data, not an official benchmark or a faithfulness conclusion.

All three downloaded base.pt files match the checkpoint SHA-256 in their paired NPZ metadata. Local copies of checkpoint.json point to those downloaded files; the received data were not edited.

Result: 12 CSV rows, 18 columns, four PNG figures and a run log.

Reproduce from the repository root after placing the paired inputs in runs/d1:

```text
python scripts/run_d1.py --scene synthetic --condition v8_az360 --runs_root runs/d1 --out_dir results/d1/phase0_provisional_2026-10-17/synthetic/v8_az360 --n_random 100
```

The provisional f_i definition requires opacity > 0.05. All three seeds' top 5% opacity-risk selections have opacity below 0.014, so their Precision@5% is necessarily zero under this label gate. Official m_i labels and the gradient_inconsistency signal are not included in this data package.
