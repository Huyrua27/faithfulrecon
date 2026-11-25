"""Checks for the snapshot's reversible visibility intervention."""
from dataclasses import replace

import torch

from frecon.camera import Camera
from frecon.eval.floater import distances, evaluate_mask, reference_depth, select_candidates
from frecon.gaussians import GaussianModel, inverse_sigmoid
from frecon.render.torch_rast import rasterize


def test_mask_recomposites_the_surface_behind_a_floater():
    # The near red splat occludes the blue splat on the reference surface.
    model = GaussianModel(
        torch.tensor([[0., 0., 1.5], [0., 0., 2.]]),
        torch.full((2, 3), 0.08).log(),
        torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        inverse_sigmoid(torch.tensor([0.95, 0.95])),
        inverse_sigmoid(torch.tensor([[1., 0., 0.], [0., 0., 1.]])),
    )
    cam = Camera(torch.eye(3), torch.zeros(3), 25, 25, 8.5, 8.5, 17, 17)
    bg = torch.ones(3)
    target = model.subset(torch.tensor([False, True])).render(rasterize, cam, bg)["image"].detach()
    cam = replace(cam, image=target)
    ref = torch.tensor([[0., 0., 2.]])
    d = distances(model.means, ref)
    depth = [reference_depth(ref, cam)]
    original = model.opacities.detach().clone()
    base = evaluate_mask(model, [cam], bg, torch.empty(0, dtype=torch.long), ref, d, depth, 0.1, 0.05)
    masked = evaluate_mask(model, [cam], bg, torch.tensor([0]), ref, d, depth, 0.1, 0.05)
    assert base[0]["w_float"] > 0
    assert masked[0]["w_float"] == 0
    assert base[0]["depth_coverage"] == 0
    assert masked[0]["depth_coverage"] == 1
    assert masked[0]["center_coverage"] == 1
    torch.testing.assert_close(masked[2][0], target)
    torch.testing.assert_close(model.opacities, original)


def test_score_threshold_does_not_spend_budget_on_unsupported_rows():
    scores = torch.tensor([0.99, 0.70, 0.70, 0.10])
    eligible = torch.tensor([False, True, True, True])
    assert select_candidates(scores, eligible, 0.75, 0.6).tolist() == [1, 2]
    assert select_candidates(scores, eligible, 0.75, 0.8).numel() == 0
