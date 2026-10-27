"""Cross-track handoff tests: identity validation, non-destructive merges, real CLI output."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from frecon.signals.second_order_export import NAMES, compute, save_scores
from scripts.export_second_order import load_checkpoint
from tests.test_second_order import tiny_context


def metadata():
    return dict(scene="synthetic", condition="v2_az360", seed=0, checkpoint_sha256="a" * 64,
                n_gaussians=20, signal_params={"opacity": {"source": "baseline"}})


def test_score_merge_and_replacement_are_safe(tmp_path):
    path = tmp_path / "scores.npz"
    baseline = np.linspace(0, 1, 20, dtype=np.float32)
    save_scores(path, {"opacity": baseline}, metadata())
    meta = metadata()
    meta["signal_params"] = {"curvature": {"eps": 1e-8}}
    save_scores(path, {"curvature": torch.arange(20).float()}, meta)
    with np.load(path, allow_pickle=True) as archive:
        np.testing.assert_array_equal(archive["opacity"], baseline)
        assert archive["curvature"].shape == (20,) and archive["curvature"].dtype == np.float32
        assert set(archive["meta"].item()["signal_params"]) == {"opacity", "curvature"}
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_scores(path, {"curvature": np.ones(20)}, meta)
    assert path.read_bytes() == before
    save_scores(path, {"curvature": np.ones(20)}, meta, overwrite=True)
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["curvature"], np.ones(20, dtype=np.float32))
        np.testing.assert_array_equal(archive["opacity"], baseline)


@pytest.mark.parametrize("change", [{"seed": 1}, {"checkpoint_sha256": "b" * 64}, {"n_gaussians": 19}])
def test_reject_mixed_runs_without_touching_archive(tmp_path, change):
    path = tmp_path / "scores.npz"
    save_scores(path, {"opacity": np.zeros(20)}, metadata())
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different checkpoint"):
        save_scores(path, {"fisher": np.zeros(20)}, dict(metadata(), **change))
    assert path.read_bytes() == before


@pytest.mark.parametrize("scores", [{"fisher": np.zeros(19)}, {"fisher": np.full(20, np.nan)}])
def test_reject_invalid_arrays(tmp_path, scores):
    path = tmp_path / "scores.npz"
    with pytest.raises(ValueError, match="finite float32"):
        save_scores(path, scores, metadata())
    assert not path.exists()


def make_run(root):
    run = root / "synthetic_sphere_smoke" / "v2_az360" / "seed_0"
    run.mkdir(parents=True)
    ctx = tiny_context()
    checkpoint = run / "base.pt"
    torch.save({"params": ctx.model.state_dict()}, checkpoint)
    manifest = dict(path="base.pt", sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), n_gaussians=20)
    (run / "checkpoint.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = root / "smoke.yaml"
    config.write_text(yaml.safe_dump(dict(device="cpu", renderer="torch", scene=dict(
        type="synthetic", objects=["sphere"], n_gt_gaussians=200, n_ref_points=200,
        resolution=16, n_train_views=2, n_test_views=2, seed=123))), encoding="utf-8")
    return run, config


def test_cli_exports_contract_and_checkpoint_guard(tmp_path):
    run, config = make_run(tmp_path)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "scripts/export_second_order.py", "--run_dir", str(run),
               "--config", str(config), "--n_samples", "4"]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(run / "scores.npz", allow_pickle=True) as archive:
        assert set(archive.files) == set(NAMES) | {"meta"}
        for name in NAMES:
            assert archive[name].shape == (20,) and archive[name].dtype == np.float32
            assert np.isfinite(archive[name]).all()
        meta = archive["meta"].item()
        assert meta["checkpoint_sha256"] == hashlib.sha256((run / "base.pt").read_bytes()).hexdigest()
        assert meta["signal_params"]["pup"]["scale_basis"] == "scale"
        assert meta["signal_params"]["s_geom"]["inconsistency"]["provider"] == "temporary_phat"
    manifest = json.loads((run / "checkpoint.json").read_text())
    manifest["sha256"] = "f" * 64
    (run / "checkpoint.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(run, "cpu")


def test_relocated_checkpoint_keeps_original_identity_guard(tmp_path):
    run, _ = make_run(tmp_path)
    moved = tmp_path / "downloaded.pt"
    (run / "base.pt").rename(moved)
    model, meta = load_checkpoint(run, "cpu", moved)
    assert model.n == 20 and meta["checkpoint_sha256"] == hashlib.sha256(moved.read_bytes()).hexdigest()
    moved.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(run, "cpu", moved)


def test_reject_different_camera_config_without_touching_archive(tmp_path):
    path = tmp_path / "scores.npz"
    meta = dict(metadata(), scene_config_sha256="a" * 64)
    save_scores(path, {"opacity": np.zeros(20)}, meta)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different scene/camera"):
        save_scores(path, {"fisher": np.zeros(20)}, dict(meta, scene_config_sha256="b" * 64))
    assert path.read_bytes() == before


def test_full_bundle_preserves_model_and_gradients():
    ctx = tiny_context()
    for cam in ctx.cams:
        cam.image = ctx.model.render(ctx.rasterize, cam, ctx.bg)["image"].detach() * 0.95
    for param in ctx.model.params.values():
        param.grad = torch.full_like(param, 0.25)
    before = {name: (p.detach().clone(), p.grad.clone()) for name, p in ctx.model.params.items()}
    compute(ctx.model, ctx.cams, ctx.rasterize, ctx.bg, n_samples=4)
    for name, param in ctx.model.params.items():
        torch.testing.assert_close(param, before[name][0], rtol=0, atol=0)
        torch.testing.assert_close(param.grad, before[name][1], rtol=0, atol=0)
