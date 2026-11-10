"""Audit the shared Phase 0/D1 package, export W4 scores, and record trained-scene evidence.

Source files remain untouched. Geometry labels are provisional; AUROC below is a handoff
sanity check, not the complete D1 protocol or evidence for D2/D3 faithfulness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.data import load_scene
from frecon.gaussians import nearest_dist
from frecon.measure import auroc, floater_labels, heldout_errors, render_views
from frecon.render import get_rasterizer
from frecon.signals.second_order_export import compute, save_scores
from scripts.export_second_order import load_checkpoint, sha256


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="runs/drive_source/data")
    parser.add_argument("--out_root", default="runs/d1")
    parser.add_argument("--results", default="results/second_order/w4/checkpoints")
    parser.add_argument("--config", default="results/phase0/config_used.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source, out, results = Path(args.source_root), Path(args.out_root), Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    rasterize, bg = get_rasterizer(cfg.get("renderer", "torch")), torch.ones(3, device=device)
    train, test, ref, _ = load_scene(cfg["scene"], rasterize, bg, device)
    config_hash = hashlib.sha256(json.dumps(cfg["scene"], sort_keys=True).encode()).hexdigest()
    audits, summaries, profiles = [], [], []
    for seed in args.seeds:
        raw_run = source / "d1/synthetic/v8_az360" / f"seed_{seed}"
        raw_seed = source / f"seed_{seed}"
        model, meta = load_checkpoint(raw_run, device, raw_seed / "base.pt")
        with np.load(raw_run / "scores.npz", allow_pickle=True) as archive:
            baseline = {k: archive[k].copy() for k in archive.files if k != "meta"}
            baseline_meta = archive["meta"].item()
        with np.load(raw_run / "geometric_labels.npz", allow_pickle=True) as archive:
            labels = {k: archive[k].copy() for k in archive.files if k != "meta"}
            label_meta = archive["meta"].item()
        for other_meta in (baseline_meta, label_meta):
            if any(other_meta.get(k) != v for k, v in meta.items()):
                raise ValueError(f"seed {seed}: mismatched archive identity")
        np.testing.assert_array_equal(labels["xyz"], model.means.detach().cpu().numpy())
        raw_signals = torch.load(raw_seed / "signals.pt", map_location="cpu", weights_only=False)
        for name, array in baseline.items():
            np.testing.assert_array_equal(array, raw_signals[name].float().numpy())
        expected = json.loads((raw_seed / "metrics.json").read_text())["err_base"]
        with torch.no_grad():
            measured = heldout_errors(render_views(model, test, rasterize, bg), test)
            distance = nearest_dist(model.means, ref)
            np.testing.assert_allclose(distance.cpu().numpy(), labels["distance"], rtol=1e-5, atol=1e-6)
            contribution = sum(model.render(rasterize, cam, bg, return_contrib=True)["contrib"] for cam in test)
            recomputed_f = floater_labels(model, ref, contribution, **label_meta["label_params"])
            np.testing.assert_array_equal(recomputed_f.cpu().numpy(), labels["f_i"])
        tolerances = {"l1": 1e-6, "dssim": 1e-5, "psnr": 1e-3}
        for key in expected:
            if abs(measured[key] - expected[key]) > tolerances[key]:
                raise ValueError(f"seed {seed}: scene/config mismatch in {key}: {measured[key]} vs {expected[key]}")
        print(f"seed {seed}: checkpoint/order/baselines/labels/scene metrics verified", flush=True)
        audits.append(dict(seed=seed, n_gaussians=model.n, checkpoint_sha256=meta["checkpoint_sha256"],
                           **{f"{key}_absolute_error": abs(measured[key] - expected[key]) for key in expected},
                           floater_fraction=float(labels["f_i"].mean()),
                           labels_provisional=label_meta["provisional"],
                           m_i_available=label_meta["m_i_available"]))
        run = out / "synthetic/v8_az360" / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoint.json", "geometric_labels.npz"):
            if (run / name).exists() and sha256(run / name) != sha256(raw_run / name):
                raise ValueError(f"refusing to replace different {run / name}")
            shutil.copy2(raw_run / name, run / name)
        if not (run / "scores.npz").exists():
            shutil.copy2(raw_run / "scores.npz", run / "scores.npz")
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        scores, options = compute(model, train, rasterize, bg, seed=seed, n_samples=args.n_samples)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        profiles.append(dict(seed=seed, n_gaussians=model.n, train_views=len(train),
                             resolution=cfg["scene"]["resolution"], n_samples=args.n_samples,
                             bundle_seconds=elapsed,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                             peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
                             gpu=torch.cuda.get_device_name() if device == "cuda" else "cpu"))
        meta.update(signal_params=options, scene_config=cfg["scene"], scene_config_sha256=config_hash)
        save_scores(run / "scores.npz", scores, meta, overwrite=args.overwrite)
        with np.load(run / "scores.npz", allow_pickle=True) as archive:
            for name, value in baseline.items():
                np.testing.assert_array_equal(archive[name], value)
            for name in archive.files:
                if name == "meta":
                    continue
                value = archive[name]
                assert value.shape == (model.n,) and value.dtype == np.float32 and np.isfinite(value).all()
                summaries.append(dict(seed=seed, signal=name, n_gaussians=model.n,
                                      min=float(value.min()), median=float(np.median(value)), max=float(value.max()),
                                      unique_scores=len(np.unique(value)),
                                      auroc_provisional_f=auroc(torch.from_numpy(value), torch.from_numpy(labels["f_i"]))))
        (results / f"seed_{seed}_options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
        print(f"seed {seed}: {len(scores)} new scores exported, baseline arrays preserved, {elapsed:.2f}s", flush=True)
    write_csv(results / "source_audit.csv", audits)
    write_csv(results / "scores_summary.csv", summaries)
    write_csv(results / "profile.csv", profiles)
    paths = list(Path("frecon/signals").glob("*.py")) + [Path(__file__), Path(args.config)]
    environment = dict(python=sys.version, torch=torch.__version__, device=device,
                       commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       command=sys.argv, source_root=str(source), scene_config_sha256=config_hash,
                       source_sha256={str(p): sha256(p) for p in paths})
    (results / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
