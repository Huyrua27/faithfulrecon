"""Train and evaluate the post-processing prototype, or score a new checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.data import load_scene  # noqa: E402
from frecon.data.synthetic import SyntheticSceneConfig, build_gt_model, make_views  # noqa: E402
from frecon.eval.floater import (  # noqa: E402
    Features, FloaterClassifier, distances, evaluate_mask, extract_features, ranking_metrics,
    reference_depth, select_candidates,
)
from frecon.gaussians import GaussianModel, inverse_sigmoid  # noqa: E402
from frecon.render.torch_rast import rasterize  # noqa: E402


@dataclass
class Run:
    entry: dict
    sha256: str
    data: Features
    d: torch.Tensor

    @property
    def split(self):
        return self.entry["split"]


def load_checkpoint(path, device):
    with path.open("rb") as stream:
        sha256 = file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        snapshot = torch.load(stream, map_location=device, weights_only=True)
    params = snapshot["params"]
    n = params["means"].shape[0]
    shapes = dict(means=(n, 3), log_scales=(n, 3), quats=(n, 4),
                  opacity_logits=(n,), color_logits=(n, 3))
    if not n or not all(key in params and params[key].shape == shape
                        and torch.isfinite(params[key]).all() for key, shape in shapes.items()):
        raise ValueError(f"{path}: expected a finite, nonempty FaithfulRecon RGB (SH0) checkpoint")
    extent = float(snapshot["extent"])
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError(f"{path}: checkpoint extent must be positive")
    return GaussianModel.from_state_dict(params, device), extent, sha256


def demo_manifest(root):
    """Build six controlled scenes by adding off-surface copies to clean snapshots."""
    variants = [("sphere",), ("box",), ("sphere", "rods"),
                ("box", "rods"), ("rods",), ("sphere", "box")]
    entries = []
    for index, (objects, split) in enumerate(zip(variants, ("train",) * 3 + ("val", "test", "test"))):
        name = "-".join(objects)
        scene = dict(type="synthetic", objects=list(objects), seed=110 + index, resolution=112,
                     n_gt_gaussians=1600, n_ref_points=16000, n_train_views=8, n_test_views=16,
                     test_elev_range=[22, 22])
        gt, ref = build_gt_model(SyntheticSceneConfig(**{k: v for k, v in scene.items() if k != "type"}))
        clean = gt.state_dict()
        rng = torch.Generator().manual_seed(900 + index)
        # Floater and surface copies share color, opacity and scale distributions.
        clean["opacity_logits"] = inverse_sigmoid(0.45 + 0.5 * torch.rand(gt.n, generator=rng))
        clean["log_scales"] += torch.rand((gt.n, 3), generator=rng) * 0.45 - 0.10
        # Sample in front, behind and between objects. Distances come from the surface, not the injection IDs.
        trial = torch.rand((gt.n * 3, 3), generator=rng) * 1.7 - 0.85
        d = distances(trial, ref)
        off_surface = trial[(d > 0.12) & (d < 0.5)][:max(1, gt.n // 7)]
        ids = torch.randint(gt.n, (len(off_surface),), generator=rng)
        added = {k: v[ids] for k, v in clean.items()}
        added["means"] = off_surface
        order = torch.randperm(gt.n + len(ids), generator=rng)
        params = {k: torch.cat((v, added[k]))[order] for k, v in clean.items()}
        checkpoint = root / f"{name}.pt"
        torch.save(dict(params=params, step=0, extent=4.4), checkpoint)
        entries.append(dict(id=name, split=split, scene=scene, checkpoint=checkpoint.name))
    manifest = root / "demo_manifest.yaml"
    manifest.write_text(yaml.safe_dump({"runs": entries}, sort_keys=False), encoding="utf-8")
    return manifest


def load_entries(manifest):
    entries = yaml.safe_load(manifest.read_text(encoding="utf-8"))["runs"]
    splits, run_ids, checkpoints = {}, set(), set()
    for entry in entries:
        run_id, split = entry["id"], entry["split"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id) or run_id in run_ids:
            raise ValueError(f"Invalid or duplicate run id: {run_id!r}")
        run_ids.add(run_id)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {run_id}: {split!r}")
        scene = entry["scene"]
        kind = scene["type"]
        for key in ("root", "stl_ply", "ref_cache", "ref_points_path"):
            if scene.get(key):
                scene[key] = str((manifest.parent / scene[key]).resolve())
        if kind == "synthetic":
            identity = tuple(sorted(set(scene.get("objects", ("sphere", "box", "rods")))))
        else:
            identity = Path(scene["root"]).name.casefold()
        prior = splits.setdefault((kind, identity), split)
        if prior != split:
            raise ValueError(f"Scene {identity!r} occurs in both {prior!r} and {split!r}")
        entry["checkpoint"] = (manifest.parent / entry["checkpoint"]).resolve()
        if entry["checkpoint"] in checkpoints:
            raise ValueError(f"Checkpoint path repeated: {entry['checkpoint']}")
        checkpoints.add(entry["checkpoint"])
    missing = {"train", "val", "test"} - {entry["split"] for entry in entries}
    if missing:
        raise ValueError(f"Manifest needs these splits: {sorted(missing)}")
    return entries


def load_inputs(entry, bg, device, *, scoring_only=False):
    model, extent, sha256 = load_checkpoint(entry["checkpoint"], device)
    scene = dict(entry["scene"])
    if scoring_only:
        if scene["type"] == "synthetic":
            scene.pop("type")
            scene["n_test_views"] = 1
            train, _ = make_views(SyntheticSceneConfig(**scene))
            if not train:
                raise ValueError("Expected at least one training camera")
            return model, extent, sha256, [cam.to(device) for cam in train], [], None
        for key in ("stl_ply", "ref_cache", "ref_points_path"):
            scene.pop(key, None)
        scene["n_test_views"] = 0
    train, test, ref, _ = load_scene(scene, rasterize, bg, device)
    if not train or (not scoring_only and (not test or ref is None or not len(ref))):
        raise ValueError(f"{entry['id']}: expected training cameras, separate test cameras and a reference surface")
    if not scoring_only and {cam.name for cam in train}.intersection(cam.name for cam in test):
        raise ValueError(f"{entry['id']}: overlapping training and test cameras")
    return model, extent, sha256, train, test, ref


def fit(runs, variant, tau, epochs, seed, device):
    torch.manual_seed(seed)
    network = FloaterClassifier(variant).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3)
    train = [r for r in runs if r.split == "train"]
    val = [r for r in runs if r.split == "val"]
    rng = np.random.default_rng(seed)
    positives = sum(int((r.d > tau).sum()) for r in train)
    total = sum(r.d.numel() for r in train)
    if not 0 < positives < total:
        raise ValueError("Training distance labels must contain both classes")
    weight = torch.tensor(min((total - positives) / positives, 8), device=device)
    best = math.inf
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(train)):
            run = train[index]
            loss = F.binary_cross_entropy_with_logits(network(run.data), (run.d > tau).float(),
                                                     pos_weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = np.mean([float(F.binary_cross_entropy_with_logits(
                network(r.data), (r.d > tau).float(), pos_weight=weight)) for r in val])
        if val_loss < best:
            best, best_epoch = float(val_loss), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    return network.eval(), dict(epoch=best_epoch, validation_loss=best)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def finite_metrics(metrics):
    return {k: v if math.isfinite(v) else None for k, v in metrics.items()}


def save_image(path, image):
    pixels = (image.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)


def write_viewer(root, cams, frames):
    root.mkdir()
    root_images = root / "frames"
    root_images.mkdir()
    for index, cam in enumerate(cams):
        save_image(root_images / f"reference-{index:02d}.png", cam.image)
        for mode, images in frames.items():
            save_image(root_images / f"{mode}-{index:02d}.png", images[index])
    template = Path(__file__).with_name("floater_pilot_viewer.html")
    viewer = template.read_text(encoding="utf-8").replace(
        "__VIEW_DATA__", json.dumps(dict(views=[c.name for c in cams], modes=list(frames))).replace("<", "\\u003c"))
    (root / "index.html").write_text(viewer, encoding="utf-8")


def compare_mask(base, result):
    base_metrics, base_views, base_rgb, base_depth = base
    metrics, per_view, rgb, depth = result
    delta = {key: metrics[key] - value for key, value in base_metrics.items()
             if value is not None and metrics[key] is not None}
    diff = (rgb - base_rgb).abs().mean(dim=-1)
    changed = diff >= 0.02
    for i, row in enumerate(per_view):
        n_ref = row["reference_pixels"]
        row["delta_psnr"] = row["psnr"] - base_views[i]["psnr"]
        row["mean_abs_rgb_change"] = float(diff[i].mean())
        row["changed_pixels"] = int(changed[i].sum())
        row["lost_depth_pixels"] = int((base_depth[i] & ~depth[i]).sum())
        row["lost_depth_fraction"] = row["lost_depth_pixels"] / n_ref if n_ref else None
    return dict(metrics=metrics, delta=delta, views=per_view,
                worst_delta_psnr=min(v["delta_psnr"] for v in per_view),
                mean_rgb_change=float(diff.mean()),
                changed_region_mean=float(diff[changed].mean()) if changed.any() else None)


def random_summary(rows):
    summary = {}
    for key in rows[0]:
        sample = [row[key] for row in rows if row[key] is not None]
        if sample:
            summary[key] = dict(mean=float(np.mean(sample)), q05=float(np.quantile(sample, 0.05)),
                                q95=float(np.quantile(sample, 0.95)))
    return summary


@torch.no_grad()
def evaluate(run, network, scores, args, bg, device, root):
    model, extent, sha256, train, test, ref = load_inputs(run.entry, bg, device)
    if sha256 != run.sha256:
        raise ValueError(f"{run.entry['id']}: checkpoint changed after feature extraction")
    gaussians = torch.arange(model.n, device=device)
    eligible = gaussians[run.data.eligible]
    empty = gaussians[:0]
    depths = [reference_depth(ref, cam) for cam in test]

    def render(ids):
        return evaluate_mask(model, test, bg, ids, ref, run.d, depths, args.tau, args.coverage_radius)

    base = render(empty)
    frames = {"base": base[2]}
    budgets, masks = [], {}
    score = scores[network]
    for budget, fraction in enumerate(args.fractions):
        ids = select_candidates(score, run.data.eligible, fraction, args.score_threshold)
        masks[f"top_{budget}"] = ids.cpu().numpy()
        intervention = render(ids)
        selected = compare_mask(base, intervention)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, budget]))
        controls = []
        for draw in range(args.random_draws):
            idx = rng.choice(len(eligible), size=len(ids), replace=False)
            rand_ids = eligible[torch.as_tensor(idx, device=device, dtype=torch.long)]
            masks[f"random_{budget}_{draw}"] = rand_ids.cpu().numpy()
            random = render(rand_ids)
            controls.append(random[0])
            if budget == 0 and draw == 0:
                frames["random"] = random[2]
        if budget == 0:
            frames["top-k"] = intervention[2]
        budgets.append(dict(fraction=fraction, k=len(ids), selected=selected,
                            random_draws=controls, random_summary=random_summary(controls)))
    probes = {}
    ordered = select_candidates(score, run.data.eligible, 1.0, args.score_threshold)
    if ordered.numel():
        peak = ordered[:1]
        separation = (model.means[ordered] - model.means[peak]).norm(dim=1)
        local = ordered[separation <= 0.12 * extent][:4]
        for mode, ids in (("single", peak), ("small-group", local),
                          ("mid-score", ordered[len(ordered) // 2:len(ordered) // 2 + 1])):
            result = render(ids)
            probes[mode] = dict(gaussian_ids=ids.tolist(), **compare_mask(base, result))
            frames[mode] = result[2]
    root.mkdir()
    np.savez_compressed(root / "scores_and_masks.npz", gaussian_id=gaussians.cpu().numpy(),
                        eligible=run.data.eligible.cpu().numpy(), distance=run.d.cpu().numpy(),
                        **{name: value.cpu().numpy() for name, value in scores.items()}, **masks)
    write_viewer(root / "viewer", test, frames)
    return dict(sha256=sha256, checkpoint=str(run.entry["checkpoint"]), n=model.n,
                eligible=len(eligible), train_views=[c.name for c in train], eval_views=[c.name for c in test],
                base=base[0], ranking={key: finite_metrics(ranking_metrics(value, run.d, args.tau))
                                      for key, value in scores.items()},
                budgets=budgets, probes=probes)


def experiment(args, device, bg):
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = demo_manifest(args.out) if args.demo else args.manifest
    entries = load_entries(manifest)
    runs, hashes = [], {}
    for entry in entries:
        model, extent, sha256, train, _, ref = load_inputs(entry, bg, device)
        if sha256 in hashes:
            raise ValueError(f"Duplicate checkpoint contents for {entry['id']} and {hashes[sha256]}")
        hashes[sha256] = entry["id"]
        print(f"Extracting {entry['split']} features for {entry['id']} ({model.n:,} Gaussians)", flush=True)
        runs.append(Run(entry, sha256, extract_features(model, train, bg, extent), distances(model.means, ref)))
    networks, training = {}, {}
    for variant in ("point", "views", "hierarchical"):
        networks[variant], training[variant] = fit(runs, variant, args.tau, args.epochs, args.seed, device)
        print(f"{variant}: chosen epoch {training[variant]['epoch']}", flush=True)
    torch.save(dict(variant="hierarchical", state_dict=networks["hierarchical"].state_dict(),
                    score_threshold=args.score_threshold, fraction=args.fractions[0]), args.out / "classifier.pt")
    report = dict(manifest=str(manifest.resolve()), synthetic_injection=args.demo, tau=args.tau,
                  coverage_radius=args.coverage_radius, score_threshold=args.score_threshold,
                  random_draws=args.random_draws, seed=args.seed, epochs=args.epochs,
                  training=training, sources=[dict(id=r.entry["id"], split=r.split, sha256=r.sha256,
                                                    checkpoint=str(r.entry["checkpoint"])) for r in runs],
                  scans={})
    for run in runs:
        if run.split != "test":
            continue
        name = run.entry["id"]
        with torch.no_grad():
            scores = {variant: model(run.data).sigmoid() for variant, model in networks.items()}
        result = evaluate(run, "hierarchical", scores, args, bg, device, args.out / name)
        report["scans"][name] = result
        print(f"{name}: evaluated {len(result['budgets'])} budgets and {len(result['probes'])} probes", flush=True)
    write_json(args.out / "metrics.json", report)
    print(f"Wrote {args.out / 'metrics.json'}", flush=True)


@torch.no_grad()
def score_only(args, device, bg):
    scene = yaml.safe_load(args.config.read_text(encoding="utf-8"))["scene"]
    entry = dict(id=args.checkpoint.stem, checkpoint=args.checkpoint, scene=scene)
    model, extent, sha256, train, _, _ = load_inputs(entry, bg, device, scoring_only=True)
    weights = torch.load(args.weights, map_location=device, weights_only=True)
    network = FloaterClassifier(weights["variant"]).to(device).eval()
    network.load_state_dict(weights["state_dict"])
    features = extract_features(model, train, bg, extent)
    scores = network(features).sigmoid()
    threshold = args.score_threshold if args.score_threshold is not None else weights["score_threshold"]
    ids = select_candidates(scores, features.eligible, weights["fraction"], threshold)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "scores_and_masks.npz", gaussian_id=np.arange(model.n),
                        hierarchical=scores.cpu().numpy(), eligible=features.eligible.cpu().numpy(),
                        candidates=ids.cpu().numpy())
    write_json(args.out / "checkpoint.json", dict(sha256=sha256, n_gaussians=model.n,
                                                  path=str(args.checkpoint.resolve()),
                                                  n_candidates=len(ids), score_threshold=threshold,
                                                  fraction=weights["fraction"],
                                                  train_views=[cam.name for cam in train]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("evaluate")
    data = run.add_mutually_exclusive_group(required=True)
    data.add_argument("--demo", action="store_true")
    data.add_argument("--manifest", type=Path)
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--tau", type=float, default=0.1)
    run.add_argument("--coverage-radius", type=float, default=0.05)
    run.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    run.add_argument("--random-draws", type=int, default=20)
    run.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--weights", type=Path, required=True)
    for cmd in (run, score):
        cmd.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        cmd.add_argument("--threads", type=int, default=4)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--score-threshold", type=float)
    args = parser.parse_args()
    if args.score_threshold is not None and not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be in [0, 1]")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "evaluate":
        if min(args.epochs, args.random_draws) < 1 or args.seed < 0:
            parser.error("epochs and random draws must be positive and seed nonnegative")
        if not all(math.isfinite(v) and v > 0 for v in (args.tau, args.coverage_radius)):
            parser.error("distances must be positive and finite")
        if not all(0 < p < 1 for p in args.fractions):
            parser.error("fractions must be in (0, 1)")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    bg = torch.ones(3, device=device)
    if args.command == "evaluate":
        experiment(args, device, bg)
    else:
        score_only(args, device, bg)


if __name__ == "__main__":
    main()
