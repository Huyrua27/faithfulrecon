"""Compute geometric labels for one D1 run from its verified base checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from frecon.benchmark.failure_labels import LabelParams, condition_labels, outcome_labels, save_labels
from frecon.benchmark.geometric_error import point_to_surface_distance
from frecon.benchmark.reference_geometry import ReferenceGeometry, check_alignment, load_reference
from frecon.benchmark.visualize_labels import plot_error, plot_label_map, plot_reference_and_gaussians
from frecon.data import load_scene
from frecon.gaussians import GaussianModel
from frecon.render import get_rasterizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("path", "sha256", "n_gaussians"):
        if key not in manifest:
            raise ValueError(f"{path} is missing required field {key!r}")
    if not isinstance(manifest["path"], str) or not manifest["path"].strip():
        raise ValueError(f"Invalid checkpoint path in {path}: {manifest['path']!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(manifest["sha256"])):
        raise ValueError(f"Invalid SHA-256 in {path}: {manifest['sha256']!r}")
    if type(manifest["n_gaussians"]) is not int or manifest["n_gaussians"] <= 0:
        raise ValueError(f"Invalid n_gaussians in {path}: {manifest['n_gaussians']!r}")
    return manifest


def validate_scores_identity(run_dir: Path, manifest: dict, scene: str,
                             condition: str, seed: int) -> None:
    """When scores are present, make sure they belong to this run/checkpoint."""
    scores_path = run_dir / "scores.npz"
    if not scores_path.is_file():
        return
    with np.load(scores_path, allow_pickle=True) as archive:
        if "meta" not in archive.files:
            raise ValueError(f"Missing metadata in {scores_path}")
        metadata = archive["meta"].item()
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata in {scores_path} must be a dictionary")
    expected = {
        "scene": scene,
        "condition": condition,
        "seed": seed,
        "checkpoint_sha256": str(manifest["sha256"]).lower(),
        "n_gaussians": int(manifest["n_gaussians"]),
    }
    for key, value in expected.items():
        actual = metadata.get(key)
        if key == "checkpoint_sha256" and isinstance(actual, str):
            actual = actual.lower()
        if actual != value:
            raise ValueError(
                f"Run identity mismatch in {scores_path}: {key}={actual!r}, expected {value!r}"
            )


def resolve_checkpoint(run_dir: Path, manifest: dict, override: Path | None) -> tuple[Path, str]:
    """Use the manifest path by default; explicit relocations must match its digest."""
    if override is not None:
        checkpoint = override.expanduser().resolve()
    else:
        checkpoint = Path(manifest["path"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (run_dir / checkpoint).resolve()
    if not checkpoint.is_file():
        hint = " Pass --checkpoint only if this is a relocated copy of the recorded checkpoint." if override is None else ""
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}.{hint}")
    actual = sha256_file(checkpoint)
    expected = str(manifest["sha256"]).lower()
    if actual.lower() != expected:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {checkpoint}: expected {expected}, got {actual}"
        )
    return checkpoint, actual


def resolve_config(run_dir: Path, condition: str, explicit: Path | None) -> Path:
    if explicit is not None:
        config = explicit
    else:
        local_candidates = (run_dir / "config_used.yaml", run_dir / "config.yaml")
        config = next((candidate for candidate in local_candidates if candidate.is_file()), None)
        if config is None:
            known = {
                "v8_az360": Path("configs/phase0.yaml"),
                "v8_az180": Path("configs/phase0_v8_az180.yaml"),
            }
            config = known.get(condition)
        if config is None:
            raise ValueError(
                f"No run-local config found for condition {condition!r}; pass --config explicitly"
            )
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config}")
    return config


def validate_condition(scene_cfg: dict, condition: str, config_path: Path) -> dict:
    """Reject a config whose camera coverage disagrees with the condition name.

    Synthetic scenes use vN_azM (views + azimuth coverage). Dataset scenes (DTU, Blender) use a
    fixed camera subset, so vN is enough and only the number of training views is checked.
    """
    synthetic = scene_cfg.get("type", "synthetic") == "synthetic"
    match = re.fullmatch(r"v(?P<views>\d+)(?:_az(?P<azimuth>\d+))?", condition)
    if match is None or (synthetic and match.group("azimuth") is None):
        expected = "v8_az360" if synthetic else "v8 or v8_az360"
        raise ValueError(
            f"Cannot verify condition {condition!r}; expected a condition like {expected}"
        )
    expected_views = int(match.group("views"))
    expected_azimuth = float(match.group("azimuth")) if match.group("azimuth") else None
    actual_views = int(scene_cfg.get("n_train_views", -1))
    if actual_views != expected_views:
        raise ValueError(
            f"Condition/config mismatch in {config_path}: {condition} encodes "
            f"{expected_views} train views, config has {actual_views}"
        )
    if synthetic:
        azimuth_range = scene_cfg.get("train_azim_range", (0.0, 360.0))
        if not isinstance(azimuth_range, (list, tuple)) or len(azimuth_range) != 2:
            raise ValueError(f"Invalid train_azim_range in {config_path}: {azimuth_range!r}")
        actual_azimuth = abs(float(azimuth_range[1]) - float(azimuth_range[0]))
        if not np.isclose(actual_azimuth, expected_azimuth, atol=1e-6):
            raise ValueError(
                f"Condition/config mismatch in {config_path}: {condition} encodes "
                f"{expected_azimuth:g}° coverage, config has {actual_azimuth:g}°"
            )
    return {"condition": condition, "n_train_views": actual_views,
            "expected_n_train_views": expected_views, "train_azim_range": scene_cfg.get("train_azim_range"),
            "expected_azimuth_coverage_deg": expected_azimuth,
            "azimuth_validation": "checked" if synthetic
            else "not_applicable_to_dataset_camera_subset"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path,
                        help="D1 run directory containing checkpoint.json")
    parser.add_argument("--checkpoint", type=Path,
                        help="explicit relocation of the manifest checkpoint; SHA-256 must match")
    parser.add_argument("--config", type=Path,
                        help="scene config; defaults to a config in run_dir or a condition-matched built-in")
    parser.add_argument("--tau", type=float, default=0.03)
    parser.add_argument("--k_views", type=int, default=2)
    parser.add_argument("--theta_deg", type=float, default=10.0)
    parser.add_argument("--min_contrib", type=float, default=1e-2)
    parser.add_argument("--min_opacity", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--figure_root", type=Path, default=Path("results/benchmark/figures"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    seed_match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    if seed_match is None:
        raise ValueError(f"Run directory must end in seed_<integer>, got {run_dir.name!r}")
    seed = int(seed_match.group(1))
    condition = run_dir.parent.name
    scene_name = run_dir.parent.parent.name

    manifest = load_manifest(run_dir)
    validate_scores_identity(run_dir, manifest, scene_name, condition, seed)
    checkpoint, digest = resolve_checkpoint(run_dir, manifest, args.checkpoint)
    config_path = resolve_config(run_dir, condition, args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("scene"), dict):
        raise ValueError(f"Expected a scene config with a 'scene' mapping: {config_path}")
    config_scene_type = cfg["scene"].get("type", "synthetic")
    if scene_name in {"synthetic", "blender", "dtu"} and scene_name != config_scene_type:
        raise ValueError(
            f"Run path identifies scene {scene_name!r}, but {config_path} configures "
            f"scene type {config_scene_type!r}"
        )
    condition_check = validate_condition(cfg["scene"], condition, config_path)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but this PyTorch installation has no CUDA")
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if "params" not in ckpt:
        raise ValueError(f"Checkpoint does not contain a 'params' state dict: {checkpoint}")
    model = GaussianModel.from_state_dict(ckpt["params"], device)
    expected_n = int(manifest["n_gaussians"])
    if model.n != expected_n:
        raise ValueError(
            f"Checkpoint Gaussian count mismatch: manifest={expected_n}, model={model.n}"
        )

    # This uses the same scene loader and full scene parameters that produced the cameras.
    train_cams, test_cams, ref_points, scene_info = load_scene(cfg["scene"], rasterize, bg, device)
    if len(train_cams) != condition_check["n_train_views"]:
        raise ValueError("Loaded camera count differs from the validated scene configuration")
    if not test_cams:
        raise ValueError("Held-out cameras are required to compute contribution-gated labels")
    if ref_points is None:
        raise ValueError(
            f"No reference geometry for scene type {cfg['scene'].get('type', 'synthetic')!r}; "
            "set scene.ref_points_path (Blender) or the DTU reference options"
        )
    if cfg["scene"].get("type", "synthetic") == "synthetic":
        # Rebuild from every SyntheticSceneConfig field (including objects, texture,
        # sampling count, and seed) so the reference matches the configured scene.
        ref = load_reference(cfg["scene"])
        if ref.points.shape != ref_points.shape or not torch.equal(ref.points.to(ref_points.device), ref_points):
            raise ValueError("Synthetic reference rebuilt from config differs from load_scene reference")
    else:
        ref = ReferenceGeometry(points=ref_points,
                                source=str(cfg["scene"].get("ref_points_path", scene_info.get("root", ""))))

    print(f"Checkpoint: {checkpoint}")
    print(f"Verified SHA-256: {digest}")
    print(f"Scene config: {config_path}")
    print(f"Condition check: {condition_check}")
    align_info = check_alignment(ref, train_cams)
    print(f"Reference frustum coverage (not an alignment verification): {align_info}")

    means = model.means.detach()
    distance = point_to_surface_distance(means, ref)
    with torch.no_grad():
        heldout_contrib = torch.zeros(model.n, device=device)
        for camera in test_cams:
            heldout_contrib += model.render(
                rasterize, camera, bg, return_contrib=True
            )["contrib"]

    params = LabelParams(k_views=args.k_views, theta_deg=args.theta_deg, tau=args.tau,
                         min_contrib=args.min_contrib, min_opacity=args.min_opacity)
    m_i = condition_labels(means, ref, train_cams, params)
    f_i = outcome_labels(distance, heldout_contrib, model.opacities.detach(), params)
    f_i_nogate = outcome_labels(distance, heldout_contrib, model.opacities.detach(), params,
                                apply_opacity_gate=False)

    meta = {
        "scene": scene_name,
        "condition": condition,
        "seed": seed,
        "checkpoint_sha256": digest,
        "n_gaussians": model.n,
    }
    output = run_dir / "geometric_labels.npz"
    save_labels(output, means.cpu(), distance.cpu(), m_i.cpu(), f_i.cpu(), meta, params,
                f_nogate=f_i_nogate.cpu())

    figure_dir = args.figure_root / condition / f"seed_{seed}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    xyz = means.cpu().numpy()
    ref_xyz = ref.points.cpu().numpy()
    m_np, f_np = m_i.cpu().numpy(), f_i.cpu().numpy()
    dist_np = distance.cpu().numpy()
    plot_reference_and_gaussians(ref_xyz, xyz, figure_dir / "ref_vs_gaussians.png")
    plot_label_map(xyz, m_np, "under_observed_m_i", figure_dir / "under_observed_map.png")
    plot_label_map(xyz, f_np, "geometric_failure_f_i", figure_dir / "geometric_failure_map.png")
    plot_error(dist_np, m_np, f_np, figure_dir / "error_distribution.png")

    summary = {
        "run_dir": str(run_dir), "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "condition_validation": condition_check,
        "seed": seed, "scene": scene_name, "condition": condition,
        "device": device, "renderer": cfg.get("renderer", "torch"),
        "n_test_views": len(test_cams), "label_params": vars(params),
        "provisional": True,
        "limitations": ["visibility proxy without depth occlusion",
                        "frustum coverage does not verify alignment",
                        "distance to sampled points approximates surface distance"],
        "distance_median": float(np.median(dist_np)),
        "distance_p95": float(np.percentile(dist_np, 95)),
        "n_gaussians": model.n, "m_i_count": int(m_i.sum()), "m_i_fraction": float(m_i.float().mean()),
        "f_i_count": int(f_i.sum()), "f_i_fraction": float(f_i.float().mean()),
        "f_i_nogate_count": int(f_i_nogate.sum()),
        "f_i_nogate_fraction": float(f_i_nogate.float().mean()),
        "reference_frustum_check": align_info,
    }
    (run_dir / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved labels: {output}")
    print(f"Saved figures: {figure_dir}")
    return summary


if __name__ == "__main__":
    main()
