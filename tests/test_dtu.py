"""DTU loader: P = K [R|t] decomposition, and end-to-end projection conventions on a fake scan."""
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from frecon.data.dtu import decompose_projection, load_dtu_views


def test_fake_scan_projection(tmp_path):
    """A world point must land on the same pixel as OpenCV projection (pixel centers at integers),
    after the loader's normalization (scale_mat) and downscaling."""
    rng = np.random.default_rng(1)
    W0, H0, ds = 160, 120, 4
    K = np.array([[300.0, 0, 79.5], [0, 300.0, 59.5], [0, 0, 1]])
    scale_mat = np.diag([2.5, 2.5, 2.5, 1.0])
    scale_mat[:3, 3] = [10.0, -4.0, 30.0]
    (tmp_path / "image").mkdir()
    (tmp_path / "mask").mkdir()
    mats = {}
    Rs, ts = [], []
    for i in range(2):
        R = Rotation.from_rotvec(rng.normal(size=3) * 0.2).as_matrix()
        t = np.array([0.0, 0.0, 40.0]) + rng.normal(size=3)
        Rs.append(R)
        ts.append(t)
        P = np.eye(4)
        P[:3, :4] = K @ np.hstack([R, t[:, None]])
        mats[f"world_mat_{i}"] = P
        mats[f"scale_mat_{i}"] = scale_mat
        Image.fromarray(np.zeros((H0, W0, 3), np.uint8)).save(tmp_path / "image" / f"{i:06d}.png")
        Image.fromarray(np.full((H0, W0), 255, np.uint8)).save(tmp_path / "mask" / f"{i:03d}.png")
    np.savez(tmp_path / "cameras_sphere.npz", **mats)

    cams, masks, sm = load_dtu_views(str(tmp_path), downscale=ds)
    x_world = np.array([10.3, -3.8, 30.4])
    x_norm = np.linalg.solve(scale_mat, np.append(x_world, 1.0))[:3]
    for cam, R, t in zip(cams, Rs, ts):
        p = K @ (R @ x_world + t)
        u_cv, v_cv = p[0] / p[2], p[1] / p[2]            # OpenCV pixel coords, full resolution
        pc = cam.R.double().numpy() @ x_norm + cam.t.double().numpy()
        u = cam.fx * pc[0] / pc[2] + cam.cx              # our continuous coords, pixel i spans [i, i+1)
        v = cam.fy * pc[1] / pc[2] + cam.cy
        assert np.isclose(u, (u_cv + 0.5) / ds, atol=1e-3)
        assert np.isclose(v, (v_cv + 0.5) / ds, atol=1e-3)
        assert cam.W == W0 // ds and cam.H == H0 // ds


def test_decompose_round_trip():
    rng = np.random.default_rng(0)
    for scale in (1.0, -3.7):
        K = np.array([[2892.3, 0.4, 823.2], [0.0, 2883.2, 619.1], [0.0, 0.0, 1.0]])
        R = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        t = rng.normal(size=3) * 100
        P = scale * K @ np.hstack([R, t[:, None]])
        K2, R2, t2 = decompose_projection(P)
        assert np.allclose(K2, K, atol=1e-6)
        assert np.allclose(R2, R, atol=1e-8)
        assert np.allclose(t2, t, atol=1e-6)
        assert np.isclose(np.linalg.det(R2), 1.0)
