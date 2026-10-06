"""Scene loading. `load_scene(cfg)` returns (train_cams, test_cams, ref_points or None, info)."""
from __future__ import annotations

import torch

from .synthetic import SyntheticSceneConfig, build_gt_model, make_views, render_gt


def load_scene(cfg: dict, rasterize, bg, device):
    kind = cfg.get("type", "synthetic")
    if kind == "synthetic":
        keys = SyntheticSceneConfig.__dataclass_fields__.keys()
        scfg = SyntheticSceneConfig(**{k: (tuple(v) if isinstance(v, list) and k.endswith("range") else v)
                                       for k, v in cfg.items() if k in keys})
        gt_model, ref = build_gt_model(scfg)
        train, test = make_views(scfg)
        train = render_gt(gt_model, train, rasterize, bg, device)
        test = render_gt(gt_model, test, rasterize, bg, device)
        return train, test, ref.to(device), {"type": "synthetic", "n_gt_gaussians": gt_model.n}
    if kind == "blender":
        from .blender import load_blender
        train, test, ref = load_blender(cfg["root"], cfg["n_train_views"], cfg["n_test_views"],
                                        cfg.get("downscale", 4), cfg.get("ref_points_path"))
        train = [c.to(device) for c in train]
        test = [c.to(device) for c in test]
        return train, test, (None if ref is None else ref.to(device)), {"type": "blender", "root": cfg["root"]}
    raise ValueError(f"unknown scene type: {kind}")
