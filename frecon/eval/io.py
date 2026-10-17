"""Load and validate paired D1 scores and geometric labels.

The file layout and metadata contract are defined in docs/data_format.md.
Validation raises on mismatches and returns warnings for provisional labels,
unavailable m_i, degenerate labels, and unverified checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunData:
    scene: str
    condition: str
    seed: int
    scores: dict[str, np.ndarray]  # signal -> (N,) float32; higher = less reliable
    labels: dict[str, np.ndarray]  # m_i/f_i: bool; distance: float32
    meta: dict = field(default_factory=dict)
    xyz: np.ndarray | None = None


def load_run(run_dir: str | Path) -> RunData:
    run_dir = Path(run_dir)
    scores_path = run_dir / "scores.npz"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing file: {scores_path}")

    labels_path = run_dir / "geometric_labels.npz"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    scores = {}

    with np.load(scores_path, allow_pickle=True) as data:
        if "meta" not in data.files:
            raise ValueError(f"Missing meta in {scores_path}")

        scores_meta = data["meta"].item()

        if not isinstance(scores_meta, dict):
            raise ValueError("scores metadata must be a dictionary")

        for feature in data.files:
            if feature == "meta":
                continue
            scores[feature] = data[feature]

    labels = {}
    with np.load(labels_path, allow_pickle=True) as data:
        if "meta" not in data.files:
            raise ValueError(f"Missing meta in {labels_path}")

        if "xyz" not in data.files:
            raise ValueError(f"Missing required array: xyz in {labels_path}")

        labels_meta = data["meta"].item()

        if not isinstance(labels_meta, dict):
            raise ValueError("labels metadata must be a dictionary")

        xyz = data["xyz"]

        for key in data.files:
            if key not in ("distance", "m_i", "f_i"):
                continue
            labels[key] = data[key]

    scene = run_dir.parent.parent.name
    condition = run_dir.parent.name
    seed_name = run_dir.name
    if not seed_name.startswith("seed_"):
        raise ValueError(f"Seed name dont start with seed_")
    if not seed_name[5:].isdigit():
        raise ValueError("Seed isnt a number")
    seed = int(seed_name[5:])

    meta = {
        "scores": scores_meta,
        "labels": labels_meta,
        "run_dir": str(run_dir),
    }
    return RunData(
        scene=scene,
        condition=condition,
        seed=seed,
        scores=scores,
        labels=labels,
        meta=meta,
        xyz=xyz,
    )


def validate(run: RunData) -> list[str]:
    """Raise on contract violations; return a list of warnings."""
    warnings = []
    scores_meta = run.meta["scores"]
    labels_meta = run.meta["labels"]

    if not isinstance(scores_meta.get("signal_params"), dict):
        raise ValueError("Missing or invalid signal_params")

    if not isinstance(labels_meta.get("label_params"), dict):
        raise ValueError("Missing or invalid label_params")

    required_fields = (
        "scene",
        "condition",
        "seed",
        "checkpoint_sha256",
        "n_gaussians",
    )
    required_labels = ("distance", "f_i", "m_i")
    for key in required_fields:
        if key not in scores_meta or key not in labels_meta:
            raise ValueError(f"Missing required metadata field: {key}")

        if scores_meta[key] != labels_meta[key]:
            raise ValueError(
                f"Metadata mismatch for {key}: "
                f"scores={scores_meta[key]!r}, "
                f"labels={labels_meta[key]!r}"
            )

    for key in ("scene", "condition", "seed"):
        expected = getattr(run, key)
        if expected != scores_meta[key]:
            raise ValueError(
                f"Metadata-to-path mismatch for {key}: "
                f"run={expected!r}, scores={scores_meta[key]!r}"
            )

    for key in required_labels:
        if key not in run.labels:
            raise ValueError(f"Missing required label: {key}")

    N = scores_meta["n_gaussians"]

    if (
        isinstance(N, (bool, np.bool_))
        or not isinstance(N, (int, np.integer))
        or N <= 0
    ):
        raise ValueError(f"Invalid n_gaussians: {N!r}")

    if run.xyz is None:
        raise ValueError("Missing xyz array")

    if run.xyz.shape != (N, 3):
        raise ValueError(f"Invalid xyz shape: expected {(N, 3)}, got {run.xyz.shape}")

    if run.xyz.dtype != np.float32:
        raise ValueError(f"Invalid xyz dtype: expected float32, got {run.xyz.dtype}")

    if not np.isfinite(run.xyz).all():
        raise ValueError("Non-finite values in xyz")

    if not run.scores:
        raise ValueError("No score signals found")

    for name, arr in run.scores.items():
        if arr.shape != (N,):
            raise ValueError(
                f"Invalid shape for {name}: expected {(N,)}, got {arr.shape}"
            )

    for name, arr in run.labels.items():
        if arr.shape != (N,):
            raise ValueError(
                f"Invalid shape for {name}: expected {(N,)}, got {arr.shape}"
            )

    for name, arr in run.scores.items():
        if arr.dtype != np.float32:
            raise ValueError(
                f"Invalid dtype for {name}: expected float32, got {arr.dtype}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in score: {name}")

    for name, arr in run.labels.items():
        if name == "distance":
            if arr.dtype != np.float32:
                raise ValueError(
                    f"Invalid dtype for {name}: expected float32, got {arr.dtype}"
                )

            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in label: {name}")
        elif name in ("m_i", "f_i"):
            if arr.dtype != np.bool_:
                raise ValueError(
                    f"Invalid dtype for {name}: expected bool, got {arr.dtype}"
                )

    if "m_i_available" not in labels_meta:
        raise ValueError("Missing required metadata field: m_i_available")
    if "provisional" not in labels_meta:
        raise ValueError("Missing required metadata field: provisional")

    available = labels_meta["m_i_available"]
    provisional = labels_meta["provisional"]

    if not isinstance(available, (bool, np.bool_)):
        raise ValueError(f"m_i_available must be bool, got {type(available).__name__}")
    if not isinstance(provisional, (bool, np.bool_)):
        raise ValueError(f"provisional must be bool, got {type(provisional).__name__}")

    if not available:
        if run.labels["m_i"].any():
            raise ValueError("m_i must be all False when m_i_available is False")
        warnings.append("m_i unavailable; skip m_i metrics")
    if provisional:
        warnings.append("provisional geometric labels; not official ground truth")

    f_prevalence = run.labels["f_i"].mean()

    if f_prevalence == 0 or f_prevalence == 1:
        warnings.append("Degenerate prevalence for f_i")

    if available:
        m_prevalence = run.labels["m_i"].mean()

        if m_prevalence == 0 or m_prevalence == 1:
            warnings.append("Degenerate prevalence for m_i")

    run_dir = Path(run.meta["run_dir"])
    checkpoint_path = run_dir / "checkpoint.json"

    if checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint_meta = json.load(f)

        if not isinstance(checkpoint_meta, dict):
            raise ValueError("Checkpoint metadata must be a dictionary")

        required_fields = ("path", "sha256", "n_gaussians")

        for key in required_fields:
            if key not in checkpoint_meta:
                raise ValueError(f"Missing checkpoint field: {key}")

        if checkpoint_meta["sha256"] != scores_meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint metadata SHA-256 mismatch")

        if checkpoint_meta["n_gaussians"] != N:
            raise ValueError("Checkpoint Gaussian count mismatch")

        path_value = checkpoint_meta["path"]

        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Invalid checkpoint path: {path_value!r}")

        checkpoint_file = Path(path_value)

        if not checkpoint_file.is_absolute():
            checkpoint_file = run_dir / checkpoint_file

        if checkpoint_file.is_file():
            hasher = hashlib.sha256()

            with checkpoint_file.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    hasher.update(chunk)

            actual_sha256 = hasher.hexdigest()

            if actual_sha256 != checkpoint_meta["sha256"]:
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch: "
                    f"expected={checkpoint_meta['sha256']}, "
                    f"actual={actual_sha256}"
                )
        else:
            warnings.append(
                f"Checkpoint file missing; SHA-256 not verified: {checkpoint_file}"
            )
    else:
        warnings.append("checkpoint.json missing; checkpoint not verified")

    return warnings


def discover_runs(
    root: str | Path,
    scene: str | None = None,
    condition: str | None = None,
) -> list[Path]:
    root = Path(root)

    candidates = root.glob("*/*/seed_*")
    runs = []
    for path in candidates:
        if not path.is_dir():
            continue
        if scene is not None and path.parent.parent.name != scene:
            continue
        if condition is not None and path.parent.name != condition:
            continue
        sd = path.name
        if not (sd.startswith("seed_") and sd[5:].isdigit()):
            continue
        runs.append(path)

    return sorted(runs)
