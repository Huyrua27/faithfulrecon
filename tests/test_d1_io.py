"""Synthetic checks for loading and validating a paired D1 run."""

import hashlib
import json

import numpy as np
import pytest

from frecon.eval.io import discover_runs, load_run, validate


@pytest.fixture
def run_dir(tmp_path):
    path = tmp_path / "runs" / "d1" / "synthetic" / "v8_az360" / "seed_0"
    path.mkdir(parents=True)
    checkpoint = path / "base.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    meta = {
        "scene": "synthetic",
        "condition": "v8_az360",
        "seed": 0,
        "checkpoint_sha256": digest,
        "n_gaussians": 3,
    }
    np.savez_compressed(
        path / "scores.npz",
        opacity=np.array([0.9, 0.2, 0.7], dtype=np.float32),
        meta=np.array({**meta, "signal_params": {}}, dtype=object),
    )
    np.savez_compressed(
        path / "geometric_labels.npz",
        xyz=np.zeros((3, 3), dtype=np.float32),
        distance=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        m_i=np.zeros(3, dtype=bool),
        f_i=np.array([True, False, True], dtype=bool),
        meta=np.array(
            {**meta, "label_params": {}, "m_i_available": False, "provisional": True},
            dtype=object,
        ),
    )
    (path / "checkpoint.json").write_text(
        json.dumps({"path": str(checkpoint), "sha256": digest, "n_gaussians": 3}),
        encoding="utf-8",
    )
    return path


def test_load_and_validate_provisional_run(run_dir):
    run = load_run(run_dir)

    assert (run.scene, run.condition, run.seed) == ("synthetic", "v8_az360", 0)
    assert set(run.scores) == {"opacity"}
    assert set(run.labels) == {"distance", "m_i", "f_i"}
    warnings = validate(run)
    assert any("m_i unavailable" in warning for warning in warnings)
    assert any("provisional" in warning for warning in warnings)


def test_mismatched_score_and_label_lengths_are_rejected(run_dir):
    run = load_run(run_dir)
    run.labels["distance"] = run.labels["distance"][:-1]
    with pytest.raises(ValueError, match="Invalid shape for distance"):
        validate(run)


def test_metadata_mismatch_is_rejected(run_dir):
    run = load_run(run_dir)
    run.meta["labels"]["seed"] = 1
    with pytest.raises(ValueError, match="Metadata mismatch for seed"):
        validate(run)


def test_unavailable_m_i_must_be_a_false_placeholder(run_dir):
    run = load_run(run_dir)
    run.labels["m_i"][0] = True
    with pytest.raises(ValueError, match="m_i"):
        validate(run)


def test_checkpoint_content_hash_is_verified(run_dir):
    run = load_run(run_dir)
    (run_dir / "base.pt").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        validate(run)


def test_discover_runs_filters_scene_and_condition(run_dir):
    root = run_dir.parents[2]
    assert discover_runs(root, scene="synthetic", condition="v8_az360") == [run_dir]
    assert discover_runs(root, scene="other") == []
