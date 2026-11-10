"""Portable handoff checks: relocation, row identity, preserved baselines and archive integrity."""
import hashlib
import json
import shutil
import zipfile

import numpy as np
import pytest
import yaml

from frecon.signals.second_order_export import NAMES, save_scores
from scripts.export_second_order import load_checkpoint
from scripts.package_second_order_handoff import BASELINES, build_package, verify_archive
from tests.test_second_order_export import make_run


@pytest.fixture
def package_inputs(tmp_path):
    runs_root = tmp_path / "runs"
    run, config = make_run(runs_root)
    model, meta = load_checkpoint(run, "cpu")
    cfg = yaml.safe_load(config.read_text())
    meta.update(scene_config=cfg["scene"],
                scene_config_sha256=hashlib.sha256(json.dumps(cfg["scene"], sort_keys=True).encode()).hexdigest(),
                signal_params={name: {"higher": "less_reliable"} for name in NAMES})
    scores = {name: np.arange(model.n, dtype=np.float32) for name in (*BASELINES, *NAMES)}
    save_scores(run / "scores.npz", scores, meta)
    baseline_root = tmp_path / "baseline"
    baseline_run = baseline_root / run.parent.parent.name / run.parent.name / run.name
    baseline_run.mkdir(parents=True)
    save_scores(baseline_run / "scores.npz", {name: scores[name] for name in BASELINES}, meta)
    labels = dict(xyz=model.means.detach().numpy(), distance=np.ones(model.n, dtype=np.float32),
                  f_i=np.arange(model.n) % 2 == 0, m_i=np.zeros(model.n, dtype=bool),
                  meta=np.array(dict(meta, label_params={}, provisional=True, m_i_available=False), dtype=object))
    np.savez_compressed(run / "geometric_labels.npz", **labels)
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "seed_0").mkdir(parents=True)
    shutil.copy2(run / "base.pt", checkpoint_root / "seed_0/base.pt")
    args = dict(runs_root=runs_root, checkpoint_root=checkpoint_root, baseline_root=baseline_root,
                config=config, out_dir=tmp_path / "package", scene=meta["scene"], condition=meta["condition"], seeds=(0,))
    return args, run, baseline_run, labels


def test_package_is_portable_and_preserves_sources(package_inputs):
    args, run, _, _ = package_inputs
    before = {name: (run / name).read_bytes() for name in ("checkpoint.json", "scores.npz", "geometric_labels.npz")}
    archive = build_package(**args)
    manifest = verify_archive(archive)
    portable_run = args["out_dir"] / "d1" / args["scene"] / args["condition"] / "seed_0"
    assert json.loads((portable_run / "checkpoint.json").read_text())["path"] == "base.pt"
    model, meta = load_checkpoint(portable_run, "cpu")  # works without a path override
    assert model.n == 20 and meta["checkpoint_sha256"] == manifest["runs"][0]["checkpoint_sha256"]
    assert set(manifest["runs"][0]["signals"]) == set(BASELINES) | set(NAMES)
    for name, contents in before.items():
        assert (run / name).read_bytes() == contents
    with pytest.raises(FileExistsError):
        build_package(**args)


def test_package_rejects_corrupted_archive(package_inputs, tmp_path):
    args, _, _, _ = package_inputs
    archive = build_package(**args)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(broken, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"changed" if name.endswith("scores.npz") else source.read(name))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(broken)


def test_package_rejects_reordered_labels(package_inputs):
    args, run, _, labels = package_inputs
    labels["xyz"] = labels["xyz"][::-1]
    np.savez_compressed(run / "geometric_labels.npz", **labels)
    with pytest.raises(AssertionError, match="label row order"):
        build_package(**args)
    assert not args["out_dir"].exists()


def test_package_rejects_unavailable_nonempty_m_i(package_inputs):
    args, run, _, labels = package_inputs
    labels["m_i"][0] = True
    np.savez_compressed(run / "geometric_labels.npz", **labels)
    with pytest.raises(ValueError, match="unavailable m_i"):
        build_package(**args)


def test_package_rejects_changed_baselines(package_inputs):
    args, _, baseline_run, _ = package_inputs
    with np.load(baseline_run / "scores.npz", allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["opacity"] = np.zeros(20, dtype=np.float32)
    np.savez_compressed(baseline_run / "scores.npz", **arrays)
    with pytest.raises(AssertionError, match="changed baseline"):
        build_package(**args)
