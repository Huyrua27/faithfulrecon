#!/usr/bin/env bash
# Protocol v2.1 batch: extra seeds + more random controls on v8_az360, new condition v8_az180,
# and D1 run dirs for both. Every step is cached; re-running resumes. Log: runs/d2v21_all.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
py() { python -W ignore "$@"; }

echo "== [1/7] base models seeds 3-4, v8_az360"
py scripts/phase0.py --config configs/phase0.yaml --seeds 3 4 --base_only
echo "== [2/7] D2 v2.1 on v8_az360 (seeds 0-4, 5 random)"
py scripts/d2_v2.py --config configs/d2v21_v8_az360.yaml
echo "== [3/7] analysis v8_az360"
py scripts/analyze_d2.py --run_dir runs/d2v21_v8_az360 > runs/d2v21_v8_az360/analyze.txt
echo "== [4/7] base models v8_az180 (seeds 0-2)"
py scripts/phase0.py --config configs/phase0_v8_az180.yaml --base_only
echo "== [5/7] D2 v2.1 on v8_az180"
py scripts/d2_v2.py --config configs/d2v21_v8_az180.yaml
echo "== [6/7] analysis v8_az180"
py scripts/analyze_d2.py --run_dir runs/d2v21_v8_az180 > runs/d2v21_v8_az180/analyze.txt
echo "== [7/7] D1 run dirs"
py scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 3 4 --scene synthetic --condition v8_az360
py scripts/prepare_d1_run.py --phase0_dir runs/v8_az180 --seeds 0 1 2 --scene synthetic --condition v8_az180
echo "== all done"
