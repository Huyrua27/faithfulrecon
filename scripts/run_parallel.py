"""Run a seed-based script (phase0.py, d2_v2.py) as several processes on one GPU.

Seeds are split round-robin over `--workers` processes; each process loads the scene once and runs
its seeds one after the other. Runs are independent and cached, so a large GPU is used by running
many small re-optimizations at the same time. Logs: <out_dir>/parallel_w<k>.log

Usage:
  python scripts/run_parallel.py --script scripts/phase0.py --config configs/dtu_scan24_v8.yaml --workers 6 --base_only
  python scripts/run_parallel.py --script scripts/d2_v2.py  --config configs/d2v21_dtu_scan24_v8.yaml --workers 6
Unknown arguments are passed through to the script.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args, passthrough = ap.parse_known_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds else cfg["seeds"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    groups = [seeds[k::args.workers] for k in range(args.workers)]
    groups = [g for g in groups if g]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs = []
    for k, g in enumerate(groups):
        log = open(out / f"parallel_w{k}.log", "a", encoding="utf-8")
        cmd = [sys.executable, "-W", "ignore", args.script, "--config", args.config,
               "--seeds", *map(str, g), *passthrough]
        print(f"worker {k}: seeds {g}")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, g))
    t0 = time.time()
    failed = []
    for p, log, g in procs:
        if p.wait() != 0:
            failed.append(g)
        log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min" + (f"; FAILED seeds: {failed} (see logs)" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
