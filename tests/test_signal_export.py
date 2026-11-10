"""Export contract, stable ties, identity rejection and CLI reproduction."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.export_signals import ranking, selected_ids, validate_scores, write_outputs


def test_ties_and_topk_are_nested():
    score = np.array([0, 4, 4, 2, 4, 1], dtype=np.float32)
    assert ranking(score).tolist() == [1, 2, 4, 3, 5, 0]
    assert selected_ids(score, 0.01).tolist() == [1]
    assert selected_ids(score, 0.5).tolist() == [1, 2, 4]
    assert selected_ids(score, 1).size == 6


@pytest.mark.parametrize("score", [np.zeros(3, dtype=np.float64),
    np.array([0, 1, np.nan], dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])
def test_bad_scores_rejected(score):
    with pytest.raises(ValueError):
        validate_scores({"test": score}, 3)


def test_csv_keeps_gaussian_order_and_topk_matches_ranking(tmp_path):
    score = np.arange(100, dtype=np.float32)
    meta = dict(scene="test", condition="v4", seed=1, n_gaussians=100,
                checkpoint_sha256="abc", signal_params={})
    write_outputs(tmp_path, {"opacity": score}, np.zeros((100, 3)), meta, {})
    with (tmp_path / "signal_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100
    assert [int(r["gaussian_id"]) for r in rows] == list(range(100))
    assert [float(r["score"]) for r in rows] == score.tolist()
    with np.load(tmp_path / "topk.npz") as z:
        assert z["opacity_top5"].tolist() == [99, 98, 97, 96, 95]
        assert z["opacity_top1"].size == 1
        assert z["opacity_top10"].size == 10


def test_cached_cli_and_hash_rejection(tmp_path):
    run = tmp_path / "input" / "synthetic" / "v8_az360" / "seed_0"
    run.mkdir(parents=True)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"opaque checkpoint bytes; cached export does not deserialize torch")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run / "checkpoint.json").write_text(json.dumps(dict(path="stale/base.pt", sha256=digest, n_gaussians=10)))
    meta = dict(scene="synthetic", condition="v8_az360", seed=0, n_gaussians=10,
                checkpoint_sha256=digest, signal_params={})
    np.savez(run / "scores.npz", opacity=np.arange(10, dtype=np.float32), meta=np.array(meta, dtype=object))
    np.savez(run / "geometric_labels.npz", xyz=np.zeros((10, 3), dtype=np.float32),
             meta=np.array(meta, dtype=object))
    script = Path(__file__).resolve().parents[1] / "scripts/export_signals.py"
    output = tmp_path / "output"
    command = [sys.executable, str(script), "--run_dir", str(run), "--checkpoint", str(checkpoint),
               "--source", "cached", "--out_dir", str(output), "--signals", "opacity", "--no_figures"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "signal_results.csv").exists()
    # An explicit checkpoint override is still identity-checked, not silently trusted.
    checkpoint.write_bytes(b"different checkpoint")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "SHA256" in result.stderr
