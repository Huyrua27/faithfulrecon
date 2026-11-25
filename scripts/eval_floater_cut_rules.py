"""Compare reversible cut rules on existing Phase 0 checkpoints, using a frozen classifier."""
from __future__ import annotations

import argparse
import json
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
import yaml

# The pilot entry point adds the repository root to sys.path.
from run_floater_pilot import compare_mask, finite_metrics, load_checkpoint, random_summary, write_json

from frecon.data import load_scene
from frecon.eval.floater import (
    FloaterClassifier, distances, evaluate_mask, extract_features, ranking_metrics,
    reference_depth, select_candidates,
)
from frecon.render.torch_rast import rasterize


GEOMETRY_TAU = 0.1
COVERAGE_RADIUS = 0.05
LIMITS = {
    "center_coverage": -0.01, "depth_coverage": -0.01,
    "psnr": -0.20, "worst_delta_psnr": -0.50,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="The Phase 0 config used to train 3DGS")
    parser.add_argument("--checkpoints", type=Path, required=True, help="Parent of seed_<n>/base.pt")
    parser.add_argument("--provenance", type=Path, required=True,
                        help="Parent of seed_<n>/checkpoint.json with expected SHA-256")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.005])
    parser.add_argument("--random-draws", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20261125)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or args.random_seed < 0:
        parser.error("seeds must be nonnegative and checkpoint seeds must be unique")
    if len(set(args.fractions)) != len(args.fractions) or not all(0 < f < 1 for f in args.fractions):
        parser.error("fractions must be unique and in (0, 1)")
    if min(args.random_draws, args.threads) < 1:
        parser.error("random draws and threads must be positive")
    return args


def screen(result, random_mean):
    delta = result["delta"]
    return {
        "w_float_below_original": delta["w_float"] < 0,
        "w_float_below_random_mean": result["metrics"]["w_float"] < random_mean,
        **{key: delta.get(key, float("-inf")) >= limit for key, limit in LIMITS.items()
           if key != "worst_delta_psnr"},
        "worst_delta_psnr": result["worst_delta_psnr"] >= LIMITS["worst_delta_psnr"],
    }


def evaluate_budget(fraction, base, scores, sizes, eligible, random_orders, render, far):
    ranked_ids = select_candidates(scores, eligible, fraction)
    size_ids = select_candidates(sizes, eligible, fraction)
    k = ranked_ids.numel()
    if k == 0:
        raise ValueError("No Gaussians contribute in the training views")
    masks = {"classifier": ranked_ids, "largest": size_ids}
    ranked = {name: compare_mask(base, render(mask)) for name, mask in masks.items()}
    random = []
    random_masks = random_orders[:, :k]
    for mask in random_masks:
        metrics, views, _, _ = render(mask)
        random.append(dict(**metrics, worst_delta_psnr=min(
            a["psnr"] - b["psnr"] for a, b in zip(views, base[1], strict=True)),
            selected_far_fraction=float(far[mask].float().mean())))
    summary = random_summary(random)
    random_mean = summary["w_float"]["mean"]
    for name, result in ranked.items():
        result["selected_far_fraction"] = float(far[masks[name]].float().mean())
        result["criteria"] = screen(result, random_mean)
        result["passes"] = all(result["criteria"].values())
        result["random_draws_beaten_in_w_float"] = sum(
            result["metrics"]["w_float"] < draw["w_float"] for draw in random)
    return (dict(fraction=fraction, k=k, **ranked, random_draws=random, random_summary=summary),
            {**masks, "random": random_masks})


@torch.inference_mode()
def evaluate_seed(seed, args, net, scene):
    train, views, ref, bg, ref_depths = scene
    directory = f"seed_{seed}"
    path = args.checkpoints / directory / "base.pt"
    expected = json.loads((args.provenance / directory / "checkpoint.json").read_text(encoding="utf-8"))
    model, extent, sha = load_checkpoint(path, "cpu")
    if sha != expected["sha256"] or model.n != expected["n_gaussians"]:
        raise ValueError(f"{path}: checkpoint disagrees with Phase 0 provenance")
    print(f"{directory}: verified {model.n:,} rows, extracting features", flush=True)
    features = extract_features(model, train, bg, extent)
    scores = net(features).sigmoid()
    d = distances(model.means, ref)
    sizes = model.scales.amax(dim=1)
    eligible_ids = features.eligible.nonzero(as_tuple=True)[0].numpy()
    rng = np.random.default_rng(np.random.SeedSequence([args.random_seed, seed]))
    max_k = min(len(eligible_ids), max(max(1, round(f * model.n)) for f in args.fractions))
    random_orders = torch.as_tensor(np.stack([
        rng.choice(eligible_ids, max_k, replace=False) for _ in range(args.random_draws)]))

    def render(ids):
        return evaluate_mask(model, views, bg, ids, ref, d, ref_depths, GEOMETRY_TAU, COVERAGE_RADIUS)

    base = render(torch.empty(0, dtype=torch.long))
    print(f"{directory}: original W_float={base[0]['w_float']:.6f}", flush=True)
    rows, masks = [], {}
    for i, fraction in enumerate(args.fractions):
        budget, ids = evaluate_budget(fraction, base, scores, sizes, features.eligible, random_orders,
                                      render, d > GEOMETRY_TAU)
        rows.append(budget)
        masks.update({f"budget_{i}_{name}": mask.numpy() for name, mask in ids.items()})
        print(f"{directory} ({100*fraction:g}%, K={budget['k']}): "
              f"classifier W_float={budget['classifier']['metrics']['w_float']:.6f}, "
              f"largest={budget['largest']['metrics']['w_float']:.6f}, "
              f"random={budget['random_summary']['w_float']['mean']:.6f}", flush=True)
    np.savez_compressed(args.out / f"{directory}_masks.npz", eligible=features.eligible.numpy(),
                        scores=scores.numpy(), **masks)
    result = dict(seed=seed, checkpoint=str(path.resolve()), sha256=sha, n=model.n, eligible=len(eligible_ids),
                  original=base[0], original_views=base[1],
                  ranking={name: finite_metrics(ranking_metrics(s, d, GEOMETRY_TAU))
                           for name, s in (("classifier", scores), ("largest", sizes))},
                  budgets=rows)
    write_json(args.out / f"{directory}.json", result)
    return result


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    weights_sha = digest(args.weights)
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    net = FloaterClassifier(weights["variant"]).eval()
    net.load_state_dict(weights["state_dict"])
    args.out.mkdir(parents=True, exist_ok=False)
    bg = torch.ones(3)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["scene"]
    train, views, ref, _ = load_scene(config, rasterize, bg, "cpu")
    if ref is None or len(ref) == 0 or not train or not views:
        raise ValueError("Expected training views, held-out views and a reference point cloud")
    scene = train, views, ref, bg, [reference_depth(ref, cam) for cam in views]
    results = [evaluate_seed(seed, args, net, scene) for seed in args.seeds]
    write_json(args.out / "metrics.json", dict(
        protocol="Phase 0 cut rules with frozen classifier",
        weights=str(args.weights.resolve()), weights_sha256=weights_sha,
        config=str(args.config.resolve()), config_sha256=digest(args.config),
        train_views=[cam.name for cam in train], heldout_views=[cam.name for cam in views],
        geometry_tau=GEOMETRY_TAU, coverage_radius=COVERAGE_RADIUS, limits=LIMITS,
        random_seed=args.random_seed, random_draws=args.random_draws, results=results))
    print(f"Wrote {args.out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
