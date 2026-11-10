"""Benchmark checks owned by Truong Thinh; no D1 evaluator changes."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from frecon.benchmark.failure_labels import LabelParams, condition_labels, outcome_labels, save_labels
from frecon.benchmark.geometric_error import point_to_surface_distance
from frecon.benchmark.reference_geometry import ReferenceGeometry, load_reference
from frecon.camera import orbit_cameras
from frecon.data.synthetic import SyntheticSceneConfig, build_gt_model
from scripts.make_labels import load_manifest, resolve_checkpoint, sha256_file, validate_condition, validate_scores_identity


def test_distance_on_sphere():
    points = torch.tensor([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                           [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    means = torch.tensor([[1.25, 0, 0], [0, .75, 0], [0, 0, 0.]])
    actual = point_to_surface_distance(means, ReferenceGeometry(points), 1, 2)
    torch.testing.assert_close(actual, torch.tensor([.25, .25, 1.]))


def test_condition_view_angle_and_nearest_mapping():
    ref = ReferenceGeometry(torch.tensor([[0., 0, 0], [.1, 0, 0]]))
    cams = orbit_cameras(2, 4, 32, 32, 60, azim_range=(0, 90))
    means = torch.tensor([[.01, 0, 0], [.09, 0, 0]])
    params = LabelParams(k_views=2, theta_deg=10)
    assert not condition_labels(means, ref, cams, params, 1, 1).any()
    assert condition_labels(means, ref, cams[:1], params).all()
    assert condition_labels(means, ref, [cams[0], cams[0]], params).all()
    assert condition_labels(means, ref, cams, replace(params, k_views=3)).all()
    # A Gaussian may change m_i when its nearest reference sample changes.
    torch.testing.assert_close(condition_labels(means, ref, cams, params),
                               condition_labels(means + .001, ref, cams, params))


def test_condition_normals_control_visibility():
    cams = orbit_cameras(2, 4, 32, 32, 60, azim_range=(0, 90))
    ref = ReferenceGeometry(torch.zeros(1, 3), normals=torch.tensor([[0., 0, -1.]]))
    assert condition_labels(ref.points, ref, cams, LabelParams()).item()


def test_outcome_label_thresholds():
    params = LabelParams(tau=.03, min_contrib=.01, min_opacity=.05)
    d = torch.tensor([.03, .04, .04, .04, .10])
    c = torch.tensor([1., .01, 1., 1., 1.])
    o = torch.tensor([1., 1., .05, .06, .8])
    actual = outcome_labels(d, c, o, params)
    torch.testing.assert_close(actual, torch.tensor([False, False, False, True, True]))
    larger_tau = outcome_labels(d, c, o, replace(params, tau=.05))
    assert not (larger_tau & ~actual).any()


def test_labels_file_roundtrip(tmp_path):
    path = tmp_path / 'geometric_labels.npz'
    meta = dict(scene='synthetic', condition='v8_az360', seed=0,
                checkpoint_sha256='a' * 64, n_gaussians=2)
    save_labels(path, np.zeros((2, 3)), [.01, .10], [True, False], [False, True], meta, LabelParams())
    with np.load(path, allow_pickle=True) as data:
        assert set(data.files) == {'xyz', 'distance', 'm_i', 'f_i', 'meta'}
        assert data['xyz'].dtype == data['distance'].dtype == np.float32
        assert data['m_i'].dtype == data['f_i'].dtype == np.bool_
        assert data['f_i'].tolist() == [False, True]
        assert data['meta'].item()['checkpoint_sha256'] == meta['checkpoint_sha256']
        assert data['meta'].item()['provisional'] is True
        assert data['meta'].item()['m_i_available'] is True


@pytest.mark.parametrize('kwargs', [{'tau': -1}, {'theta_deg': float('nan')},
                                   {'k_views': 1.5}, {'min_opacity': 2}])
def test_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        LabelParams(**kwargs)


def test_nan_geometry_rejected():
    with pytest.raises(ValueError, match='finite'):
        point_to_surface_distance(torch.tensor([[float('nan'), 0, 0]]),
                                  ReferenceGeometry(torch.zeros(1, 3)))


def test_checkpoint_relocation_hash_and_no_fallback(tmp_path):
    ckpt = tmp_path / 'relocated.pt'
    ckpt.write_bytes(b'exact checkpoint bytes')
    manifest = dict(path=str(tmp_path / 'missing.pt'), sha256=sha256_file(ckpt), n_gaussians=2)
    assert resolve_checkpoint(tmp_path, manifest, ckpt)[1] == manifest['sha256']
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path, manifest, None)
    ckpt.write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        resolve_checkpoint(tmp_path, manifest, ckpt)


def test_manifest_gaussian_count_must_be_integer(tmp_path):
    (tmp_path / 'checkpoint.json').write_text(json.dumps(dict(path='a.pt', sha256='a'*64, n_gaussians=2.5)))
    with pytest.raises(ValueError, match='n_gaussians'):
        load_manifest(tmp_path)


def test_condition_config_mismatch():
    cfg = dict(type='synthetic', n_train_views=8, train_azim_range=[0, 360])
    validate_condition(cfg, 'v8_az360', 'config.yaml')
    for condition in ('v8_az180', 'v4_az360'):
        with pytest.raises(ValueError, match='mismatch'):
            validate_condition(cfg, condition, 'config.yaml')


def test_scores_wrong_seed_rejected(tmp_path):
    meta = dict(scene='synthetic', condition='v8_az360', seed=1,
                checkpoint_sha256='a'*64, n_gaussians=2)
    np.savez(tmp_path / 'scores.npz', meta=np.array(meta, dtype=object))
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_scores_identity(tmp_path, dict(sha256='a'*64, n_gaussians=2),
                                 'synthetic', 'v8_az360', 0)


def test_synthetic_full_config_matches_scene_reference():
    cfg = dict(type='synthetic', n_ref_points=80, n_gt_gaussians=80,
               objects=['sphere'], seed=17, texture_freq=3.)
    ref = load_reference(cfg)
    _, expected = build_gt_model(SyntheticSceneConfig(**{k: v for k, v in cfg.items() if k != 'type'}))
    torch.testing.assert_close(ref.points, expected, rtol=0, atol=0)


def test_make_labels_small_scene_end_to_end(tmp_path):
    from scripts.make_labels import main

    scene = dict(type='synthetic', resolution=12, n_ref_points=32, n_gt_gaussians=24,
                 objects=['sphere'], n_train_views=2, n_test_views=1,
                 train_azim_range=[0, 360])
    model, _ = build_gt_model(SyntheticSceneConfig(**{k: v for k, v in scene.items() if k != 'type'}))
    checkpoint = tmp_path / 'base.pt'
    torch.save({'params': model.state_dict()}, checkpoint)
    run_dir = tmp_path / 'synthetic/v2_az360/seed_0'
    run_dir.mkdir(parents=True)
    (run_dir / 'checkpoint.json').write_text(json.dumps(dict(
        path=str(checkpoint), sha256=sha256_file(checkpoint), n_gaussians=model.n)))
    config = tmp_path / 'config.yaml'
    config.write_text(json.dumps(dict(scene=scene, renderer='torch')))
    figures = tmp_path / 'figures'
    result = main(['--run_dir', str(run_dir), '--config', str(config),
                   '--device', 'cpu', '--figure_root', str(figures)])
    assert result['n_gaussians'] == model.n and result['provisional'] is True
    assert (run_dir / 'geometric_labels.npz').is_file()
    assert (run_dir / 'label_summary.json').is_file()
    assert len(list((figures / 'v2_az360/seed_0').glob('*.png'))) == 4


def test_dataset_condition_accepts_view_count_only():
    cfg = dict(type='dtu', n_train_views=8)
    assert validate_condition(cfg, 'v8', 'config.yaml')['azimuth_validation'] == 'not_applicable_to_dataset_camera_subset'
    validate_condition(cfg, 'v8_az360', 'config.yaml')
    with pytest.raises(ValueError, match='mismatch'):
        validate_condition(cfg, 'v4', 'config.yaml')
    with pytest.raises(ValueError, match='Cannot verify'):
        validate_condition(dict(type='synthetic', n_train_views=8), 'v8', 'config.yaml')


def test_labels_file_with_nogate(tmp_path):
    path = tmp_path / 'geometric_labels.npz'
    meta = dict(scene='dtu_scan24', condition='v8', seed=0, checkpoint_sha256='a' * 64, n_gaussians=3)
    save_labels(path, np.zeros((3, 3)), [.01, .10, .2], [False] * 3, [False, True, False], meta,
                LabelParams(), f_nogate=[False, True, True])
    with np.load(path, allow_pickle=True) as data:
        assert data['f_i_nogate'].dtype == np.bool_
        assert data['f_i_nogate'].tolist() == [False, True, True]
        assert data['meta'].item()['f_i_nogate_available'] is True
    with pytest.raises(ValueError, match='subset'):
        save_labels(path, np.zeros((3, 3)), [.01, .10, .2], [False] * 3, [True, True, False], meta,
                    LabelParams(), f_nogate=[False, True, True])
