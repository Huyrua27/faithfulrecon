#!/usr/bin/env bash
# Follow-up to run_d2v21_all.sh: add seed 5 on v8_az360 so that D2 has 6 paired observations
# (a two-sided Wilcoxon signed-rank test cannot reach p < 0.05 with 5 pairs: min p = 2/2^5).
# Waits for the main batch to finish, then trains seed 5, runs D2 v2.1 on seeds 0-5 (cached runs
# are reused), re-analyzes, and prepares the D1 run dir. Log: runs/seed5_followup.log
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
py() { python -W ignore "$@"; }

echo "waiting for run_d2v21_all.sh ..."
idle=0
until grep -q "^== all done" runs/d2v21_all.log 2>/dev/null; do
  if tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python; then idle=0; else idle=$((idle + 1)); fi
  if [ "$idle" -ge 3 ]; then echo "main batch stopped without finishing — not starting seed 5"; exit 1; fi
  sleep 60
done
set -e
echo "== [1/4] base model seed 5, v8_az360"
py scripts/phase0.py --config configs/phase0.yaml --seeds 5 --base_only
echo "== [2/4] D2 v2.1 on v8_az360, seeds 0-5"
py scripts/d2_v2.py --config configs/d2v21_v8_az360.yaml --seeds 0 1 2 3 4 5
echo "== [3/4] analysis v8_az360 (6 seeds)"
py scripts/analyze_d2.py --run_dir runs/d2v21_v8_az360 > runs/d2v21_v8_az360/analyze.txt
echo "== [4/4] D1 run dir seed 5"
py scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 5 --scene synthetic --condition v8_az360
echo "== all done"
