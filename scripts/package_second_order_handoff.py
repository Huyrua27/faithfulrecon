"""Build and verify a portable W4 package; checkpoints/arrays stay outside Git.

python scripts/package_second_order_handoff.py --out_dir runs/handoff/w4-2026-10-27
python scripts/package_second_order_handoff.py --verify runs/handoff/w4-2026-10-27.zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.signals.second_order_export import NAMES
from scripts.export_second_order import load_checkpoint, sha256

BASELINES = ("random", "opacity", "gradmag", "visibility")


def validate_run(run, checkpoint, baseline_run):
    """Check identity, row alignment and metadata before copying any source artifact."""
    model, identity = load_checkpoint(run, "cpu", checkpoint)
    with np.load(run / "scores.npz", allow_pickle=True) as scores, np.load(
        run / "geometric_labels.npz", allow_pickle=True
    ) as labels, np.load(baseline_run / "scores.npz", allow_pickle=True) as baseline:
        score_meta, label_meta = scores["meta"].item(), labels["meta"].item()
        for meta in (score_meta, label_meta, baseline["meta"].item()):
            if any(meta.get(key) != value for key, value in identity.items()):
                raise ValueError(f"mismatched archive/checkpoint identity: {run}")
        required = set(NAMES) | set(BASELINES)
        if not required.issubset(scores.files):
            raise ValueError(f"missing handoff signals: {required - set(scores.files)}")
        for name in scores.files:
            if name == "meta":
                continue
            value = scores[name]
            if value.shape != (model.n,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid score contract: {name}")
        for name in BASELINES:
            np.testing.assert_array_equal(scores[name], baseline[name], err_msg=f"changed baseline: {name}")
        options = score_meta.get("signal_params", {})
        if not score_meta.get("scene_config_sha256") or "scene_config" not in score_meta:
            raise ValueError("missing scene configuration provenance")
        for name in NAMES:
            if options.get(name, {}).get("higher") != "less_reliable":
                raise ValueError(f"missing score orientation/options: {name}")
        for key in ("provisional", "m_i_available"):
            if not isinstance(label_meta.get(key), (bool, np.bool_)):
                raise ValueError(f"missing/invalid label status: {key}")
        for name, shape, dtype in (("xyz", (model.n, 3), np.float32),
                                   ("distance", (model.n,), np.float32),
                                   ("f_i", (model.n,), np.bool_), ("m_i", (model.n,), np.bool_)):
            if labels[name].shape != shape or labels[name].dtype != dtype or not np.isfinite(labels[name]).all():
                raise ValueError(f"invalid label contract: {name}")
        np.testing.assert_array_equal(labels["xyz"], model.means.detach().numpy(), err_msg="label row order changed")
        if not label_meta["m_i_available"] and labels["m_i"].any():
            raise ValueError("unavailable m_i must be all False")
        return dict(identity, signals=sorted(required), signal_params=options,
                    provisional=bool(label_meta["provisional"]),
                    m_i_available=bool(label_meta["m_i_available"]))


def verify_archive(path):
    """Read every member back and verify the exact manifest file set and SHA256s."""
    import hashlib
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        manifest = json.loads(archive.read("manifest.json"))
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("archive file set differs from manifest")
        for name, record in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name or ":" in name:
                raise ValueError(f"unsafe archive path: {name}")
            contents = archive.read(name)
            if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError(f"archive checksum mismatch: {name}")
    return manifest


def build_package(runs_root, checkpoint_root, baseline_root, config, out_dir,
                  scene="synthetic", condition="v8_az360", seeds=(0, 1, 2)):
    runs_root, checkpoint_root, baseline_root, config, out_dir = map(
        Path, (runs_root, checkpoint_root, baseline_root, config, out_dir))
    archive_path = out_dir.with_suffix(".zip")
    if out_dir.exists() or archive_path.exists():
        raise FileExistsError("package output already exists; choose a new output path")
    records = []
    for seed in seeds:
        run = runs_root / scene / condition / f"seed_{seed}"
        checkpoint = checkpoint_root / f"seed_{seed}" / "base.pt"
        baseline_run = baseline_root / scene / condition / f"seed_{seed}"
        records.append(validate_run(run, checkpoint, baseline_run))
    import hashlib
    import yaml
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(json.dumps(cfg["scene"], sort_keys=True).encode()).hexdigest()
    for seed in seeds:
        with np.load(runs_root / scene / condition / f"seed_{seed}" / "scores.npz", allow_pickle=True) as scores:
            if scores["meta"].item()["scene_config_sha256"] != config_hash:
                raise ValueError("package config differs from score scene configuration")
    out_dir.mkdir(parents=True)
    shutil.copy2(config, out_dir / "config.yaml")
    for seed in seeds:
        run = runs_root / scene / condition / f"seed_{seed}"
        target = out_dir / "d1" / scene / condition / f"seed_{seed}"
        target.mkdir(parents=True)
        for name in ("scores.npz", "geometric_labels.npz"):
            shutil.copy2(run / name, target / name)
        shutil.copy2(checkpoint_root / f"seed_{seed}" / "base.pt", target / "base.pt")
        shutil.copy2(run / "checkpoint.json", target / "checkpoint.source.json")
        manifest = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
        manifest["path"] = "base.pt"
        (target / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# W4 second-order handoff\n\n"
        "Extract this ZIP into a new directory. Verify it before use with:\n\n"
        "    python scripts/package_second_order_handoff.py --verify <package.zip>\n\n"
        "Use the repository's merged D1 implementation:\n\n"
        f"    python scripts/run_d1.py --scene {scene} --condition {condition} "
        "--runs_root <extracted-package>/d1 --out_dir <results-dir> --n_random 100\n\n"
        "All six new scores use high = less reliable. Four baseline arrays are preserved.\n"
        "Checkpoints are included and referenced relative to each run; original manifests\n"
        "are preserved in checkpoint.source.json. Do not copy those source paths over\n"
        "the portable manifests. config.yaml reproduces the source scene.\n\n"
        "Labels are provisional; m_i_available=False means skip m_i metrics. The v provider,\n"
        "insufficient-view fallback, counts and all options are recorded in scores.npz/meta\n"
        "and manifest.json. Do not combine packages with different providers as one signal\n"
        "definition. This package does not claim\n"
        "D2/D3 validity. GTX 1650 profiling and teammate scientific review remain open.\n",
        encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    source_paths = list((root / "frecon/signals").glob("*.py")) + [Path(__file__), root / "scripts/export_second_order.py"]
    manifest = dict(schema="faithfulrecon_w4_handoff_v1", scene=scene, condition=condition,
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    source_sha256={p.relative_to(root).as_posix(): sha256(p) for p in source_paths},
                    scene_config_sha256=config_hash, runs=records,
                    files={p.relative_to(out_dir).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
                           for p in sorted(out_dir.rglob("*")) if p.is_file()})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir).as_posix())
    verify_archive(archive_path)
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", help="Verify an existing ZIP instead of creating a package")
    parser.add_argument("--runs_root", default="runs/d1")
    parser.add_argument("--checkpoint_root", default="runs/drive_source/data")
    parser.add_argument("--baseline_root", default="runs/drive_source/data/d1")
    parser.add_argument("--config", default="results/phase0/config_used.yaml")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_archive(args.verify)
        print(f"verified {len(manifest['runs'])} runs and {len(manifest['files'])} files")
    else:
        if not args.out_dir:
            parser.error("--out_dir is required to create a package")
        path = build_package(args.runs_root, args.checkpoint_root, args.baseline_root, args.config, args.out_dir)
        print(f"verified package: {path}, SHA256={sha256(path)}")


if __name__ == "__main__":
    main()
