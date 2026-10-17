"""Compute geometric labels for a checkpoint.                     Owner: Trường Thịnh"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np
import torch

from frecon.data import load_scene
from frecon.render import get_rasterizer
from frecon.train import TrainConfig, Trainer
from frecon.benchmark.reference_geometry import load_reference, check_alignment
from frecon.benchmark.geometric_error import point_to_surface_distance
from frecon.benchmark.failure_labels import LabelParams, condition_labels, outcome_labels, save_labels
from frecon.benchmark.visualize_labels import plot_reference_and_gaussians, plot_label_map, plot_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/d1/synthetic/v8_az360/seed_0")
    ap.add_argument("--config", default="configs/phase0.yaml")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--k_views", type=int, default=2)
    ap.add_argument("--theta_deg", type=float, default=10.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=dev)

    # 1. Đọc thông tin checkpoint và tìm base.pt an toàn
    ckpt_info = {}
    if (run_dir / "checkpoint.json").exists():
        ckpt_info = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))

    ckpt_path = run_dir / "base.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(f"runs/phase0/{run_dir.name}/base.pt")

    print(f"Đang nạp checkpoint từ: {ckpt_path}")
    train_cams, test_cams, _, _ = load_scene(cfg["scene"], rasterize, bg, dev)
    model = Trainer.load(str(ckpt_path), TrainConfig(**cfg["train"]), rasterize, bg, dev).model

    # 2. Sanity check frustum coverage
    ref = load_reference(cfg["scene"])
    align_info = check_alignment(ref, train_cams, rasterize)
    print(f"Frustum coverage info: {align_info}")

    # 3. Tính continuous error d_i (dùng double-chunking cdist)
    means = model.means.detach()
    distance = point_to_surface_distance(means, ref)

    # 4. Tính held-out contribution trên test views
    with torch.no_grad():
        contrib = torch.zeros(model.n, device=dev)
        for c in test_cams:
            contrib += model.render(rasterize, c, bg, return_contrib=True)["contrib"]

    # 5. Sinh nhãn m_i và f_i
    params = LabelParams(k_views=args.k_views, theta_deg=args.theta_deg, tau=args.tau)
    m_i = condition_labels(means, ref, train_cams, params)
    f_i = outcome_labels(distance, contrib, model.opacities.detach(), params)

    print(f"Computed labels: N={model.n}, m_i rate={float(m_i.float().mean()):.2%}, f_i rate={float(f_i.float().mean()):.2%}")

    # 6. Lưu file geometric_labels.npz chính thức với đầy đủ metadata
    meta = {
        "scene": cfg["scene"].get("type", "synthetic"),
        "condition": run_dir.parent.name,
        "seed": int(run_dir.name.split("_")[-1]),
        "checkpoint_sha256": ckpt_info.get("sha256", ""),
        "n_gaussians": model.n,
        "n_train_views": len(train_cams),
        "n_test_views": len(test_cams),
        "heldout_contrib_aggregation": "sum_over_test_views",
    }
    save_labels(run_dir / "geometric_labels.npz", means.cpu(), distance.cpu(), m_i.cpu(), f_i.cpu(), meta, params)
    print(f"Saved labels to {run_dir / 'geometric_labels.npz'}")

    # 7. Xuất Figures vào thư mục riêng của từng seed (không bị ghi đè)
    fig_dir = Path("results/benchmark/figures") / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_reference_and_gaussians(ref.points.cpu().numpy(), means.cpu().numpy(), fig_dir / "ref_vs_gaussians.png")
    plot_label_map(means.cpu().numpy(), m_i.cpu().numpy(), "under_observed_m_i", fig_dir / "under_observed_map.png")
    plot_label_map(means.cpu().numpy(), f_i.cpu().numpy(), "geometric_failure_f_i", fig_dir / "geometric_failure_map.png")
    plot_error(distance.cpu().numpy(), m_i.cpu().numpy(), f_i.cpu().numpy(), fig_dir / "error_distribution.png")
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
