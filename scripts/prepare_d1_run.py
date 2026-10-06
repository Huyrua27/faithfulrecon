"""Create a D1 run directory (docs/data_format.md §2) from an existing Phase 0 seed, so that
the three tasks can start on real data before their own exporters exist.

Writes runs/d1/<scene>/<condition>/seed_<s>/:
  checkpoint.json         path + sha256 + N of base.pt
  scores.npz              Phase 0 signals (random, opacity, gradmag, visibility)
  geometric_labels.npz    PROVISIONAL: distance + f_i from the Phase 0 code, no m_i

Usage:
  python scripts/prepare_d1_run.py --phase0_dir runs/phase0 --seeds 0 1 2 \
      --scene synthetic --condition v8_az360
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.data import load_scene  # noqa: E402
from frecon.gaussians import nearest_dist  # noqa: E402
from frecon.measure import floater_labels  # noqa: E402
from frecon.render import get_rasterizer  # noqa: E402
from frecon.train import TrainConfig, Trainer  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0_dir", default="runs/phase0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scene", default="synthetic")
    ap.add_argument("--condition", default="v8_az360")
    ap.add_argument("--out_root", default="runs/d1")
    args = ap.parse_args()

    p0 = Path(args.phase0_dir)
    cfg = yaml.safe_load((p0 / "config.yaml").read_text(encoding="utf-8"))
    dev = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=dev)
    _, test_cams, ref_pts, _ = load_scene(cfg["scene"], rasterize, bg, dev)
    tau = cfg["labels"]["tau"]

    for seed in args.seeds:
        sdir = p0 / f"seed_{seed}"
        ckpt = sdir / "base.pt"
        model = Trainer.load(ckpt, TrainConfig(**cfg["train"]), rasterize, bg, dev).model
        n = model.n
        digest = sha256(ckpt)
        meta = {"scene": args.scene, "condition": args.condition, "seed": seed,
                "checkpoint_sha256": digest, "n_gaussians": n}

        out = Path(args.out_root) / args.scene / args.condition / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.json").write_text(json.dumps(
            {"path": str(ckpt.resolve()), "sha256": digest, "n_gaussians": n, "source": "phase0"}, indent=2))

        sig = torch.load(sdir / "signals.pt", weights_only=False)
        scores = {k: v.float().cpu().numpy() for k, v in sig.items() if not k.startswith("_")}
        assert all(v.shape == (n,) for v in scores.values()), "signals.pt does not match the checkpoint"
        np.savez_compressed(out / "scores.npz", **scores,
                            meta=np.array(dict(meta, signal_params={"source": "phase0 signals.pt"}), dtype=object))

        with torch.no_grad():
            contrib = torch.zeros(n, device=dev)
            for c in test_cams:
                contrib += model.render(rasterize, c, bg, return_contrib=True)["contrib"]
            dist = nearest_dist(model.means.detach(), ref_pts)
            f = floater_labels(model, ref_pts, contrib, tau)
        np.savez_compressed(
            out / "geometric_labels.npz",
            xyz=model.means.detach().cpu().numpy().astype(np.float32),
            distance=dist.cpu().numpy().astype(np.float32),
            m_i=np.zeros(n, dtype=bool), f_i=f.cpu().numpy(),
            meta=np.array(dict(meta, provisional=True, m_i_available=False,
                               label_params={"tau": tau, "min_opacity": 0.05, "min_contrib": 1e-2}), dtype=object))
        print(f"seed {seed}: N={n}, f_i prevalence={float(f.float().mean()):.3%} -> {out}")


if __name__ == "__main__":
    main()
