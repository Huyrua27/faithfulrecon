"""Export Phát's signals to a D1 run without changing the other track's exporter.

python scripts/export_second_order.py --run_dir runs/d1/synthetic/v8_az360/seed_0 \
    --config configs/phase0.yaml --n_samples 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frecon.data import load_scene
from frecon.gaussians import GaussianModel
from frecon.render import get_rasterizer
from frecon.signals.second_order_export import NAMES, compute, save_scores


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(run_dir, device, checkpoint_path=None):
    run_dir = Path(run_dir)
    match = re.fullmatch(r"seed_(\d+)", run_dir.name)
    if match is None:
        raise ValueError("run_dir must follow <scene>/<condition>/seed_<integer>")
    manifest = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    path = Path(checkpoint_path if checkpoint_path is not None else manifest["path"])
    if checkpoint_path is None and not path.is_absolute():
        path = run_dir / path
    digest = sha256(path)
    if digest != manifest["sha256"]:
        raise ValueError("checkpoint SHA256 does not match checkpoint.json")
    model = GaussianModel.from_state_dict(torch.load(path, map_location=device, weights_only=False)["params"], device)
    if model.n != manifest["n_gaussians"]:
        raise ValueError("checkpoint Gaussian count does not match checkpoint.json")
    meta = dict(scene=run_dir.parent.parent.name, condition=run_dir.parent.name, seed=int(match[1]),
                checkpoint_sha256=digest, n_gaussians=model.n)
    return model, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="configs/phase0.yaml")
    parser.add_argument("--checkpoint", help="Relocated checkpoint; original manifest SHA256/count still required")
    parser.add_argument("--signals", choices=NAMES, nargs="+", default=list(NAMES))
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--fisher_eps", type=float, default=1e-12)
    parser.add_argument("--delta", type=float, default=1e-12)
    parser.add_argument("--n_min", type=int, default=2)
    parser.add_argument("--scale_basis", choices=("scale", "log_scale"), default="scale")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or (cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.run_dir, device, args.checkpoint)  # validate before scene loading
    rasterize = get_rasterizer(cfg.get("renderer", "torch"))
    bg = torch.ones(3, device=device)
    train, _, _, _ = load_scene(cfg["scene"], rasterize, bg, device)
    scores, options = compute(model, train, rasterize, bg, names=args.signals, seed=meta["seed"],
                              n_samples=args.n_samples, eps=args.eps, fisher_eps=args.fisher_eps,
                              delta=args.delta, n_min=args.n_min, scale_basis=args.scale_basis)
    meta.update(signal_params=options, scene_config=cfg["scene"],
                scene_config_sha256=hashlib.sha256(json.dumps(cfg["scene"], sort_keys=True).encode()).hexdigest())
    path = Path(args.run_dir) / "scores.npz"
    save_scores(path, scores, meta, overwrite=args.overwrite)
    for name, value in scores.items():
        print(f"{name}: N={value.numel()}, min={value.min().item():.6g}, "
              f"median={value.median().item():.6g}, max={value.max().item():.6g}")
    print(f"scores -> {path} ({device}); high = less reliable")


if __name__ == "__main__":
    main()
