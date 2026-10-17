"""3DGS optimization: Adam per parameter group, exponential position LR decay, and the
adaptive density control (clone / split / prune / opacity reset) of Kerbl et al. 2023."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from .gaussians import GaussianModel, PARAM_NAMES, inverse_sigmoid
from .losses import photometric_loss
from .render.torch_rast import quat_to_rotmat


@dataclass
class TrainConfig:
    iterations: int = 7000
    lr_means_init: float = 1.6e-4   # x scene extent
    lr_means_final: float = 1.6e-6  # x scene extent
    lr_log_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_colors: float = 2.5e-3
    lambda_dssim: float = 0.2
    densify_from: int = 500
    densify_until: int = 3500
    densify_every: int = 100
    densify_grad_thresh: float = 2e-4  # in NDC units, as in 3DGS
    percent_dense: float = 0.01
    opacity_reset_every: int = 3000
    min_opacity: float = 0.005
    max_gaussians: int = 150_000       # safety cap for small GPUs
    log_every: int = 500


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expon_lr(step: int, lr_init: float, lr_final: float, max_steps: int) -> float:
    t = min(max(step / max(max_steps, 1), 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


class Trainer:
    def __init__(self, model: GaussianModel, cfg: TrainConfig, extent: float, rasterize, bg):
        self.model = model
        self.cfg = cfg
        self.extent = extent
        self.rasterize = rasterize
        self.bg = bg
        self.step = 0
        lrs = {"means": cfg.lr_means_init * extent, "log_scales": cfg.lr_log_scales, "quats": cfg.lr_quats,
               "opacity_logits": cfg.lr_opacity, "color_logits": cfg.lr_colors}
        self.opt = torch.optim.Adam(
            [{"params": [model.params[k]], "lr": lrs[k], "name": k} for k in PARAM_NAMES], lr=0.0, eps=1e-15)
        self._reset_stats()

    # ---- statistics used by densification -----------------------------------------------
    def _reset_stats(self):
        n, dev = self.model.n, self.model.means.device
        self.grad_accum = torch.zeros(n, device=dev)
        self.grad_denom = torch.zeros(n, device=dev)

    def _set_means_lr(self, lr: float):
        for g in self.opt.param_groups:
            if g["name"] == "means":
                g["lr"] = lr

    # ---- one optimization step ------------------------------------------------------------
    def train_step(self, cam, densify: bool = True, means_lr: float | None = None):
        cfg = self.cfg
        if means_lr is None:
            means_lr = expon_lr(self.step, cfg.lr_means_init * self.extent,
                                cfg.lr_means_final * self.extent, cfg.iterations)
        self._set_means_lr(means_lr)
        out = self.model.render(self.rasterize, cam, self.bg)
        if densify:
            out["means2d"].retain_grad()
        loss = photometric_loss(out["image"], cam.image, cfg.lambda_dssim)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        if densify and self.step < cfg.densify_until:
            with torch.no_grad():
                vis = out["visible"]
                g = out["means2d"].grad
                if g is not None:
                    g_ndc = torch.stack([g[:, 0] * cam.W * 0.5, g[:, 1] * cam.H * 0.5], -1).norm(dim=-1)
                    self.grad_accum[vis] += g_ndc[vis]
                    self.grad_denom[vis] += 1
        self.opt.step()
        with torch.no_grad():  # keep quaternions well conditioned
            q = self.model.params["quats"]
            q.div_(q.norm(dim=-1, keepdim=True).clamp_min(1e-12))

        self.step += 1
        if densify and cfg.densify_from <= self.step < cfg.densify_until:
            if self.step % cfg.densify_every == 0:
                self.densify_and_prune(prune_big=self.step > cfg.opacity_reset_every)
            if self.step % cfg.opacity_reset_every == 0:
                self.reset_opacity()
        return float(loss)

    def fit(self, cams, iterations: int | None = None, densify: bool = True, seed: int = 0,
            means_lr: float | None = None, log=print, callback=None):
        """Cycle through `cams` in a seeded random order (as in 3DGS).
        `callback(it)` is called after each step with the number of steps done (1-based)."""
        rng = random.Random(seed)
        iterations = self.cfg.iterations if iterations is None else iterations
        stack, losses = [], []
        for it in range(iterations):
            if not stack:
                stack = list(range(len(cams)))
                rng.shuffle(stack)
            cam = cams[stack.pop()]
            losses.append(self.train_step(cam, densify=densify, means_lr=means_lr))
            if callback is not None:
                callback(it + 1)
            if log and self.cfg.log_every and (it + 1) % self.cfg.log_every == 0:
                log(f"  iter {it + 1:5d}/{iterations}  loss {sum(losses[-100:]) / len(losses[-100:]):.4f}"
                    f"  #G {self.model.n}")
        return losses

    def final_means_lr(self) -> float:
        return self.cfg.lr_means_final * self.extent

    # ---- structural edits (mirror 3DGS optimizer bookkeeping) -----------------------------
    def prune(self, keep: torch.Tensor):
        for g in self.opt.param_groups:
            p = g["params"][0]
            st = self.opt.state.get(p, None)
            new_p = torch.nn.Parameter(p.detach()[keep].contiguous())
            if st:
                st["exp_avg"] = st["exp_avg"][keep]
                st["exp_avg_sq"] = st["exp_avg_sq"][keep]
                del self.opt.state[p]
                self.opt.state[new_p] = st
            g["params"][0] = new_p
            self.model.params[g["name"]] = new_p
        self.grad_accum = self.grad_accum[keep]
        self.grad_denom = self.grad_denom[keep]

    def _append(self, new: dict):
        for g in self.opt.param_groups:
            p = g["params"][0]
            ext = new[g["name"]]
            st = self.opt.state.get(p, None)
            new_p = torch.nn.Parameter(torch.cat([p.detach(), ext], 0).contiguous())
            if st:
                st["exp_avg"] = torch.cat([st["exp_avg"], torch.zeros_like(ext)], 0)
                st["exp_avg_sq"] = torch.cat([st["exp_avg_sq"], torch.zeros_like(ext)], 0)
                del self.opt.state[p]
                self.opt.state[new_p] = st
            g["params"][0] = new_p
            self.model.params[g["name"]] = new_p
        n_new = next(iter(new.values())).shape[0]
        dev = self.grad_accum.device
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(n_new, device=dev)])
        self.grad_denom = torch.cat([self.grad_denom, torch.zeros(n_new, device=dev)])

    @torch.no_grad()
    def densify_and_prune(self, prune_big: bool):
        cfg, m = self.cfg, self.model
        grads = self.grad_accum / self.grad_denom.clamp_min(1)
        max_scale = m.scales.max(dim=1).values
        big = max_scale > cfg.percent_dense * self.extent
        room = cfg.max_gaussians - m.n
        if room > 0:
            clone = (grads >= cfg.densify_grad_thresh) & ~big
            split = (grads >= cfg.densify_grad_thresh) & big
            # respect the cap: drop the lowest-gradient candidates first
            n_req = int(clone.sum()) + 2 * int(split.sum())
            if n_req > room:
                cand = clone | split
                thr = torch.topk(grads[cand], max(room // 2, 1)).values.min()
                clone &= grads >= thr
                split &= grads >= thr
            n0 = m.n
            if clone.any():
                self._append({k: m.params[k].detach()[clone] for k in PARAM_NAMES})
            if split.any():
                split = torch.cat([split, torch.zeros(m.n - n0, dtype=torch.bool, device=split.device)])
                s = m.scales[split]
                R = quat_to_rotmat(m.quats[split])
                new = {}
                samples = torch.randn(2, *s.shape, device=s.device) * s[None]
                new["means"] = (torch.einsum("nij,knj->kni", R, samples) + m.means[split][None]).reshape(-1, 3)
                new["log_scales"] = torch.log(s / (0.8 * 2)).repeat(2, 1)
                for k in ("quats", "opacity_logits", "color_logits"):
                    v = m.params[k].detach()[split]
                    new[k] = v.repeat(2, *([1] * (v.dim() - 1)))
                self._append(new)
                split = torch.cat([split, torch.zeros(m.n - split.numel(), dtype=torch.bool, device=split.device)])
                self.prune(~split)

        prune = m.opacities < cfg.min_opacity
        if prune_big:
            prune |= m.scales.max(dim=1).values > 0.1 * self.extent
        if prune.any():
            self.prune(~prune)
        self._reset_stats()

    @torch.no_grad()
    def reset_opacity(self):
        m = self.model
        new = inverse_sigmoid(torch.minimum(m.opacities, torch.full_like(m.opacities, 0.01)))
        m.params["opacity_logits"].copy_(new)
        for g in self.opt.param_groups:
            if g["name"] == "opacity_logits":
                st = self.opt.state.get(g["params"][0], None)
                if st:
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()

    # ---- checkpoints ---------------------------------------------------------------------
    def save(self, path):
        opt_state = {}
        for g in self.opt.param_groups:
            st = self.opt.state.get(g["params"][0], {})
            opt_state[g["name"]] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in st.items()}
        torch.save({"params": self.model.state_dict(), "opt": opt_state, "step": self.step,
                    "extent": self.extent}, path)

    @classmethod
    def load(cls, path, cfg: TrainConfig, rasterize, bg, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        model = GaussianModel.from_state_dict(ck["params"], device)
        tr = cls(model, cfg, ck["extent"], rasterize, bg)
        for g in tr.opt.param_groups:
            st = ck["opt"].get(g["name"], {})
            if st:
                tr.opt.state[g["params"][0]] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in st.items()}
        tr.step = ck["step"]
        return tr
