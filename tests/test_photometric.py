"""Per-view photometric perturbation: deterministic, identity at zero strength, object-only."""
import numpy as np
import torch

from frecon.camera import Camera
from frecon.data.photometric import apply, perturb_views, sample_params


def _cam(img):
    return Camera(R=torch.eye(3), t=torch.zeros(3), fx=1.0, fy=1.0, cx=2.0, cy=2.0, W=4, H=4, image=img)


def test_zero_strength_is_identity():
    p = sample_params(3)[0]
    img = torch.rand(4, 4, 3)
    assert torch.equal(apply(img, torch.ones(4, 4, dtype=torch.bool), p), img)


def test_deterministic_and_seed_dependent():
    a = sample_params(5, 1.0, 0.1, 0.2, seed=0)
    assert a == sample_params(5, 1.0, 0.1, 0.2, seed=0)
    assert a != sample_params(5, 1.0, 0.1, 0.2, seed=1)
    assert all(0.5 <= p["gain"] <= 2.0 for p in a)


def test_only_inside_mask():
    img = torch.full((4, 4, 3), 0.4)
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    (cam,), (p,) = perturb_views([_cam(img)], [mask], {"exposure_stops": 1.0, "wb": 0.1, "gamma": 0.2, "seed": 3})
    assert torch.equal(cam.image[2:], img[2:])
    expected = (0.4 * p["gain"] * torch.tensor(p["wb"])).clamp(0, 1) ** p["gamma"]
    assert torch.allclose(cam.image[0, 0], expected)
