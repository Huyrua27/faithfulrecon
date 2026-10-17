"""Synthetic CLI checks for the current D1 metrics runner slice."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from frecon.eval.random_baseline import random_reference

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_run(
    root: Path,
    seed: int,
    *,
    label_seed: int | None = None,
    mi_available: bool = False,
    mi_all_false: bool = False,
) -> None:
    run_dir = root / "synthetic" / "v8_az360" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "base.pt"
    checkpoint.write_bytes(f"synthetic checkpoint {seed}".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    meta = {
        "scene": "synthetic",
        "condition": "v8_az360",
        "seed": seed,
        "checkpoint_sha256": digest,
        "n_gaussians": 4,
    }
    np.savez_compressed(
        run_dir / "scores.npz",
        opacity=np.array([0.9, 0.2, 0.7, 0.1], dtype=np.float32),
        meta=np.array({**meta, "signal_params": {}}, dtype=object),
    )
    np.savez_compressed(
        run_dir / "geometric_labels.npz",
        xyz=np.zeros((4, 3), dtype=np.float32),
        distance=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        m_i=(
            np.array([False, True, False, True], dtype=bool)
            if mi_available and not mi_all_false
            else np.zeros(4, dtype=bool)
        ),
        f_i=np.array([True, False, True, False], dtype=bool),
        meta=np.array(
            {
                **meta,
                "seed": seed if label_seed is None else label_seed,
                "label_params": {},
                "m_i_available": mi_available,
                "provisional": not mi_available,
            },
            dtype=object,
        ),
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps({"path": str(checkpoint), "sha256": digest, "n_gaussians": 4}),
        encoding="utf-8",
    )


def run_cli(
    runs_root: Path, out_dir: Path, *, n_random: int = 100
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_d1.py"),
            "--scene",
            "synthetic",
            "--condition",
            "v8_az360",
            "--runs_root",
            str(runs_root),
            "--out_dir",
            str(out_dir),
            "--n_random",
            str(n_random),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_one_fi_row_per_seed_and_skips_unavailable_mi(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0)
    make_run(runs_root, 1)
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir)

    assert result.returncode == 0, result.stderr
    with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {"0", "1"}
    assert {row["label"] for row in rows} == {"f_i"}
    assert {row["signal"] for row in rows} == {"opacity"}
    assert {
        "random_auroc_mean",
        "random_auprc_mean",
        "random_precision_at_5_mean",
    } <= set(rows[0])
    labels = np.array([True, False, True, False], dtype=bool)
    distance = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    for row in rows:
        baseline = random_reference(labels, distance, R=100, seed=int(row["seed"]))
        assert float(row["random_auroc_mean"]) == baseline["auroc"][0]
        assert float(row["random_auprc_mean"]) == baseline["auprc"][0]
        assert float(row["random_precision_at_5_mean"]) == baseline["precision_at_5"][0]
    log = (out_dir / "logs" / "run_d1.log").read_text(encoding="utf-8")
    assert "sha256=" in log
    assert "provisional geometric labels" in log
    assert "Aggregate: signal=opacity label=f_i metric=auroc" in log
    assert "n_random=100" in log
    assert "n_valid=2/2" in log
    assert "Matched random: seed=0 label=f_i metric=auroc R=100" in log
    assert "q05=" in log and "q95=" in log
    for name in (
        "roc.png",
        "pr_curve.png",
        "failure_enrichment.png",
        "score_vs_error.png",
    ):
        with Image.open(out_dir / "figures" / name) as image:
            assert image.format == "PNG"
            image.verify()
    assert not (out_dir / "figures" / "m_i").exists()


def test_cli_fails_before_output_when_a_run_has_mismatched_metadata(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0)
    make_run(runs_root, 1, label_seed=7)
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir)

    assert result.returncode != 0
    assert "Metadata mismatch for seed" in result.stderr
    assert not (out_dir / "metrics.csv").exists()


def test_cli_includes_mi_when_available(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0, mi_available=True)
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir)

    assert result.returncode == 0, result.stderr
    with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["label"] for row in rows} == {"f_i", "m_i"}
    for name in ("roc.png", "pr_curve.png", "failure_enrichment.png"):
        with Image.open(out_dir / "figures" / "m_i" / name) as image:
            assert image.format == "PNG"
            image.verify()


def test_cli_rejects_nonpositive_random_draws_before_creating_output(tmp_path):
    out_dir = tmp_path / "results"
    result = run_cli(tmp_path / "runs", out_dir, n_random=0)

    assert result.returncode != 0
    assert "--n_random must be positive" in result.stderr
    assert not out_dir.exists()


def test_cli_keeps_available_degenerate_mi_as_nan(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0, mi_available=True, mi_all_false=True)
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir)

    assert result.returncode == 0, result.stderr
    with (out_dir / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    mi_row = next(row for row in rows if row["label"] == "m_i")
    assert np.isnan(float(mi_row["auroc"]))
    assert np.isnan(float(mi_row["random_auroc_mean"]))
    log = (out_dir / "logs" / "run_d1.log").read_text(encoding="utf-8")
    assert "Degenerate prevalence for m_i" in log


def test_cli_reports_how_many_seeds_have_a_defined_metric(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0)
    make_run(runs_root, 1)
    labels_path = (
        runs_root / "synthetic" / "v8_az360" / "seed_1" / "geometric_labels.npz"
    )
    with np.load(labels_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["f_i"] = np.zeros(4, dtype=bool)
    np.savez_compressed(labels_path, **payload)
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir, n_random=3)

    assert result.returncode == 0, result.stderr
    log = (out_dir / "logs" / "run_d1.log").read_text(encoding="utf-8")
    aggregate = next(
        line
        for line in log.splitlines()
        if "Aggregate: signal=opacity label=f_i metric=auroc" in line
    )
    assert "n_valid=1/2" in aggregate
    assert "seeds=[0]" in aggregate
    assert "excluded_seeds=[1]" in log


def test_cli_rejects_duplicate_seed_identities(tmp_path):
    runs_root = tmp_path / "runs" / "d1"
    make_run(runs_root, 0)
    condition_dir = runs_root / "synthetic" / "v8_az360"
    shutil.copytree(condition_dir / "seed_0", condition_dir / "seed_00")
    out_dir = tmp_path / "results"

    result = run_cli(runs_root, out_dir, n_random=3)

    assert result.returncode != 0
    assert "Duplicate run identity" in result.stderr
    assert not (out_dir / "metrics.csv").exists()
