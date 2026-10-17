"""Export checkpoint or cached scores, rankings and Top 1/5/10% 3D figures.

Owner: Quỳnh Anh. See docs/research/QuynhAnh_signal_research.md for commands.
Cached mode needs numpy/matplotlib only; checkpoint mode also needs torch/PyYAML.
Source data is read only when --out_dir points to a separate run directory.
Export ties use ascending Gaussian ID; the Phase 0 intervention rule is untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLORS = {"opacity": "#2a78d6", "gradmag": "#eb6834", "visibility": "#1baf7a",
          "gradient_inconsistency": "#eda100", "random": "#8a8984"}
FRACTIONS = (0.01, 0.05, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scores(scores: dict, n: int) -> None:
    if n <= 0:
        raise ValueError("checkpoint must contain at least one Gaussian")
    for name, score in scores.items():
        if score.shape != (n,) or score.dtype != np.float32 or not np.isfinite(score).all():
            raise ValueError(f"{name}: expected ({n},) float32, all finite")


def ranking(score: np.ndarray) -> np.ndarray:
    """Descending raw score, exact ties ordered by ascending checkpoint row ID."""
    return np.lexsort((np.arange(score.size), -score))


def selected_ids(score: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1 or score.size == 0:
        raise ValueError("fraction must be in (0, 1] and scores nonempty")
    return ranking(score)[:max(1, round(fraction * score.size))]


def read_archive(path: Path) -> tuple[dict, dict]:
    # Object metadata is part of the project's trusted local NPZ format.
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        arrays = {k: archive[k].copy() for k in archive.files if k != "meta"}
    return arrays, meta


def check_identity(meta: dict, digest: str, n: int, description: str) -> None:
    if meta.get("checkpoint_sha256") != digest or meta.get("n_gaussians") != n:
        raise ValueError(f"{description}: checkpoint hash / Gaussian count mismatch")


def plot_topk(xyz: np.ndarray, scores: dict, meta: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    lo, hi = xyz.min(0), xyz.max(0)
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()) / 2, 1e-3) * 1.05
    prefix = f"{meta['scene']}_{meta['condition']}_seed{meta['seed']}"
    for name, score in scores.items():
        for frac in FRACTIONS:
            ids = selected_ids(score, frac)
            fig = plt.figure(figsize=(7.2, 6.4))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(*xyz.T, s=0.6, c="#9ba4af", alpha=0.10, rasterized=True,
                       depthshade=False)
            ax.scatter(*xyz[ids].T, s=5, c=COLORS.get(name, "#9a4ec4"), alpha=0.85,
                       depthshade=False, label=f"Selected: {len(ids):,} / {len(xyz):,}")
            for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                setter(value - half, value + half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-55)
            ax.set(xlabel="World X", ylabel="World Y", zlabel="World Z")
            ax.set_title(f"{name} | Top {frac:.0%}\n"
                         f"{meta['scene']} / {meta['condition']} / seed {meta['seed']}")
            ax.legend(loc="upper left", fontsize=8)
            fig.text(0.5, 0.025, "Gaussian centers; higher score = selected first. "
                     "Fixed camera and bounds within each seed.", ha="center", fontsize=8)
            fig.subplots_adjust(bottom=0.09, top=0.90, left=0.01, right=0.97)
            fig.savefig(destination / f"{prefix}_{name}_top{round(frac * 100)}.png", dpi=135)
            plt.close(fig)


def write_outputs(out: Path, scores: dict, xyz: np.ndarray, meta: dict,
                  diagnostics: dict, no_csv: bool = False) -> dict:
    n = meta["n_gaussians"]
    validate_scores(scores, n)
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError("xyz must be finite (N, 3), in checkpoint order")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", **scores, meta=np.array(meta, dtype=object))
    ranks = {name: ranking(s) for name, s in scores.items()}
    np.savez_compressed(out / "rankings.npz", **ranks,
                        meta=np.array({"tie_policy": "score descending, gaussian_id ascending",
                                       "checkpoint_sha256": meta["checkpoint_sha256"]}, dtype=object))
    if not no_csv:
        with (out / "signal_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scene", "condition", "seed", "signal", "gaussian_id", "score"])
            for name, score in scores.items():
                writer.writerows((meta["scene"], meta["condition"], meta["seed"], name, i, float(s))
                                 for i, s in enumerate(score))
    selections = {}
    summaries = {}
    for name, score in scores.items():
        summary = {"min": float(score.min()), "median": float(np.median(score)),
                   "max": float(score.max()), "n_unique": int(np.unique(score).size),
                   "n_zero": int((score == 0).sum()), "topk": {}}
        for frac in FRACTIONS:
            ids = selected_ids(score, frac)
            key = f"{name}_top{round(frac * 100)}"
            selections[key] = ids
            cutoff = score[ids[-1]]
            info = {"n_selected": len(ids), "cutoff": float(cutoff),
                    "n_tied_at_cutoff": int((score == cutoff).sum())}
            if "vis_count" in diagnostics:
                info["n_zero_visibility"] = int((diagnostics["vis_count"][ids] == 0).sum())
            if "gi_count" in diagnostics:
                info["n_gi_insufficient_views"] = int((diagnostics["gi_count"][ids] <
                                                         meta["signal_params"].get(
                                                             "gradient_inconsistency", {}).get("n_min", 2)).sum())
            summary["topk"][str(round(frac * 100))] = info
        summaries[name] = summary
    np.savez_compressed(out / "topk.npz", **selections,
                        meta=np.array({"checkpoint_sha256": meta["checkpoint_sha256"],
                                       "tie_policy": "score descending, gaussian_id ascending"}, dtype=object))
    np.savez_compressed(out / "diagnostics.npz", xyz=xyz, **diagnostics,
                        meta=np.array(meta, dtype=object))
    summary = {"meta": meta, "signals": summaries}
    if "vis_count" in diagnostics:
        summary["n_zero_visibility"] = int((diagnostics["vis_count"] == 0).sum())
    if "gi_count" in diagnostics:
        summary["n_gi_insufficient_views"] = int((diagnostics["gi_count"] <
            meta["signal_params"]["gradient_inconsistency"]["n_min"]).sum())
    (out / "signal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, type=Path, help="directory containing checkpoint.json")
    ap.add_argument("--checkpoint", type=Path, help="override stale manifest path; SHA256 must match")
    ap.add_argument("--source", choices=("checkpoint", "cached"), default="checkpoint")
    ap.add_argument("--scores", type=Path, help="cached NPZ, default <run_dir>/scores.npz")
    ap.add_argument("--labels", type=Path, help="xyz archive for cached mode")
    ap.add_argument("--out_dir", type=Path, help="default <run_dir>; use a separate path to preserve input")
    ap.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
    ap.add_argument("--figure_dir", type=Path, default=Path("runs/signals/figures"))
    ap.add_argument("--signals", nargs="+", default=["random", "opacity", "gradmag", "visibility"])
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scene")
    ap.add_argument("--condition")
    ap.add_argument("--delta", type=float, default=1e-12)
    ap.add_argument("--delta_rel", type=float, default=0.0)
    ap.add_argument("--n_min", type=int, default=2)
    ap.add_argument("--contrib_eps", type=float, default=1e-3)
    ap.add_argument("--no_csv", action="store_true")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()
    start = time.perf_counter()
    if args.threads < 1 or not np.isfinite(args.contrib_eps) or args.contrib_eps < 0:
        ap.error("threads must be positive and contrib_eps finite/nonnegative")
    if len(set(args.signals)) != len(args.signals):
        ap.error("signal names must be unique")
    manifest = json.loads((args.run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or Path(manifest["path"])
    if args.checkpoint is None and not checkpoint.is_absolute():
        checkpoint = args.run_dir / checkpoint
    if not checkpoint.is_file():
        ap.error("checkpoint path is unavailable; supply --checkpoint with the local base.pt")
    digest = sha256(checkpoint)
    n = manifest["n_gaussians"]
    if digest != manifest["sha256"]:
        raise ValueError("checkpoint SHA256 differs from manifest")
    cached_meta = {}
    diagnostics = {}
    if args.source == "cached":
        arrays, cached_meta = read_archive(args.scores or args.run_dir / "scores.npz")
        check_identity(cached_meta, digest, n, "cached scores")
        missing = set(args.signals) - arrays.keys()
        if missing:
            raise ValueError(f"cached scores lack {sorted(missing)}; use --source checkpoint")
        scores = {name: arrays[name] for name in args.signals}
        labels, label_meta = read_archive(args.labels or args.run_dir / "geometric_labels.npz")
        check_identity(label_meta, digest, n, "xyz archive")
        xyz = labels["xyz"]
        params = cached_meta.get("signal_params", {})
        compute_meta = {"score_origin": "cached", "source_scores": str(
            (args.scores or args.run_dir / "scores.npz").resolve()),
            "source_scores_sha256": sha256(args.scores or args.run_dir / "scores.npz"),
            "original_metadata": cached_meta}
    # Identity fields must agree with cached metadata; checkpoint mode inherits them
    # from an existing NPZ when available, otherwise from the documented run layout.
    elif (args.run_dir / "scores.npz").exists():
        _, cached_meta = read_archive(args.run_dir / "scores.npz")
        check_identity(cached_meta, digest, n, "run scores")
    match = re.fullmatch(r"seed_(\d+)", args.run_dir.name)
    seed = args.seed if args.seed is not None else cached_meta.get("seed", int(match[1]) if match else 0)
    scene = args.scene or cached_meta.get("scene", args.run_dir.parent.parent.name)
    condition = args.condition or cached_meta.get("condition", args.run_dir.parent.name)
    for name, value in (("seed", seed), ("scene", scene), ("condition", condition)):
        if name in cached_meta and cached_meta[name] != value:
            raise ValueError(f"{name} disagrees with existing run metadata")
    if args.source == "checkpoint":
        import torch
        import yaml
        from frecon.data import load_scene
        from frecon.gaussians import GaussianModel
        from frecon.render import get_rasterizer
        from frecon.signals import REGISTRY, SignalContext

        torch.set_num_threads(args.threads)
        torch.manual_seed(seed)
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but this PyTorch installation has no CUDA")
        rasterize = get_rasterizer(cfg.get("renderer", "torch"))
        bg = torch.ones(3, device=device)
        # No optimizer is needed for analysis; weights_only avoids unpickling arbitrary classes.
        model = GaussianModel.from_state_dict(torch.load(checkpoint, map_location=device,
                                                        weights_only=True)["params"], device)
        if model.n != n:
            raise ValueError("checkpoint Gaussian count differs from manifest")
        print(f"seed {seed}: loading training images, N={n}, device={device}", flush=True)
        cams, _, _, _ = load_scene(cfg["scene"], rasterize, bg, device)
        ctx = SignalContext(model, cams, rasterize, bg, seed=seed)
        loss_weight = cfg.get("train", {}).get("lambda_dssim", 0.2)
        params = {name: {"contrib_eps": args.contrib_eps, "lambda_dssim": loss_weight}
                  for name in args.signals if name in ("gradmag", "visibility", "gradient_inconsistency")}
        if "gradient_inconsistency" in params:
            params["gradient_inconsistency"].update(delta=args.delta, delta_rel=args.delta_rel, n_min=args.n_min)
        scores = {}
        for name in args.signals:
            print(f"seed {seed}: computing {name}", flush=True)
            if name not in REGISTRY:
                raise ValueError(f"unknown signal: {name}")
            scores[name] = REGISTRY[name](ctx, **params.get(name, {})).detach().cpu().numpy()
        xyz = model.means.detach().cpu().numpy()
        if "view_stats" in ctx.cache:
            diagnostics["vis_count"] = ctx.cache["view_stats"]["vis_count"].cpu().numpy()
        if "gradient_inconsistency_stats" in ctx.cache:
            stats = ctx.cache["gradient_inconsistency_stats"]
            diagnostics.update(gi_count=stats["count"].cpu().numpy(),
                               gi_mean_norm_sq=stats["mean"].square().sum(-1).cpu().numpy(),
                               gi_variance_trace=stats["variance_trace"].cpu().numpy())
            params["gradient_inconsistency"]["effective_delta"] = stats["effective_delta"]
            params["gradient_inconsistency"]["insufficient_views_score"] = 0.0
            params["gradient_inconsistency"]["covariance"] = "population"
        compute_meta = {"score_origin": "checkpoint", "config_path": str(args.config.resolve()),
                        "config_sha256": sha256(args.config), "scene_config": cfg["scene"],
                        "renderer": cfg.get("renderer", "torch"), "device": device,
                        "torch_version": str(torch.__version__), "n_train_views": len(cams),
                        "signal_order": args.signals, "visibility_tie_break": "legacy shared seeded RNG"}
        implementation_files = ["scripts/export_signals.py", "frecon/gaussians.py", "frecon/losses.py",
            "frecon/data/synthetic.py", "frecon/render/torch_rast.py", "frecon/render/gsplat_rast.py",
            "frecon/signals/common.py", "frecon/signals/opacity.py", "frecon/signals/gradmag.py",
            "frecon/signals/visibility.py", "frecon/signals/gradient_inconsistency.py"]
        repo = Path(__file__).resolve().parents[1]
        compute_meta["implementation_sha256"] = {name: sha256(repo / name) for name in implementation_files}
    meta = dict(scene=scene, condition=condition, seed=seed, checkpoint_sha256=digest,
                n_gaussians=n, signal_params=params, score_orientation="higher = less reliable",
                ranking_tie_policy="score descending, gaussian_id ascending", **compute_meta)
    output = args.out_dir or args.run_dir
    summary = write_outputs(output, scores, xyz, meta, diagnostics, args.no_csv)
    (output / "checkpoint.json").write_text(json.dumps(dict(path=str(checkpoint.resolve()),
        sha256=digest, n_gaussians=n, source="signal_export"), indent=2), encoding="utf-8")
    if not args.no_figures:
        plot_topk(xyz, scores, meta, args.figure_dir)
    for name, s in summary["signals"].items():
        print(f"{name}: min={s['min']:.6g}, median={s['median']:.6g}, max={s['max']:.6g}, "
              f"unique={s['n_unique']}, zeros={s['n_zero']}", flush=True)
    print(f"Exported {len(scores)} signals x {n} Gaussians to {output} "
          f"in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
