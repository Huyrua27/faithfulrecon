"""CAGS - compensation-aware Gaussian suppression inside 3DGS optimization (working name).

Per-Gaussian risk q_i = p_geo_i * sigmoid((delta_hat_i - 1) / T) is estimated every `cags_every` steps from
GT-free features computed by a probe pass over the training views, smoothed by an EMA (inherited by clone / split
children) and used by three actions:
  gate  densification: no clone / split for Gaussians with risk > cags_tau_gate
  att   opacity attenuation as a per-step cap  alpha_i <= (1 - m_i) ** gamma  (not cumulative, Adam cannot regrow it)
  mod   update-level modulation of colour / SH / scale: theta <- theta_old + (1 - m_i) (theta_adam - theta_old)
with m_i = clip((r_i - tau_s) / (1 - tau_s), 0, 1), limited to the top `cags_budget` fraction of Gaussians.

Modes (TrainConfig.cags_mode): none | log (features + labels only) | oracle_geo | oracle_cmp (risk from GT labels,
oracle diagnostics) | det (frozen detector, GT-free) | rule (hand-made score on the same features, GT-free).
GT (STL points) and the clean images are used ONLY for labels / oracle runs and for logging.

Probe: one render per training view with features [view colour (3), camera depth (1), probe channels (K)].
The probe channels are zero vectors with requires_grad; for a pixel weight map c_k, the gradient of
sum_p c_k(p) R_k(p) w.r.t. probe_i,k is sum_p c_k(p) w_ip (w_ip = blending weight), i.e. footprint-weighted
statistics of any per-pixel map, for every Gaussian, from a single backward.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .gaussians import inverse_sigmoid, nearest_dist
from .render.torch_rast import quat_to_rotmat

# ---- feature table: (name, group, bounded in [0, 1]?) ------------------------------------------------------------
FEATS = [
    ("opacity", "A", True), ("ls_max", "A", False), ("ls_min", "A", False), ("aniso", "A", False),
    ("col_r", "A", True), ("col_g", "A", True), ("col_b", "A", True), ("sh_energy", "A", False),
    ("dist_center", "A", False), ("depth_ratio", "A", False),
    ("n_vis", "G", True), ("entropy", "G", True), ("max_share", "G", True), ("front_frac", "G", True),
    ("log_contrib", "G", False), ("knn_dist", "G", False), ("knn_opacity", "G", True), ("knn_front", "G", True),
    ("knn_share", "G", True),
    ("res_abs", "P", False), ("res_sgn_std", "P", False), ("conf_dc", "P", True), ("conf_sh", "P", True),
    ("knn_conf", "P", True),
    ("age", "D", True), ("gen", "D", False), ("d_opacity", "D", False), ("d_ls", "D", False),
    ("grad_acc", "D", False), ("parent_risk", "D", True), ("t_norm", "D", True),
]
FEAT_NAMES = [f[0] for f in FEATS]
GROUPS = {g: [i for i, f in enumerate(FEATS) if f[1] == g] for g in "AGPD"}
N_PROBE = 6          # probe channels: 1, |res|, signed log res, log phi (3)
EPS = 1e-8


# ---- per-Gaussian auxiliary state (kept in sync with clone / split / prune) --------------------------------------
def _aux_new(tr, n, par=None):
    dev = tr.model.means.device
    a = tr._aux
    if par is not None and len(par) == n and len(a["birth"]):
        return {"birth": torch.full((n,), float(tr.step), device=dev), "gen": a["gen"][par] + 1,
                "op_last": a["op_last"][par], "ls_last": a["ls_last"][par], "risk": a["risk"][par],
                "parent_risk": a["risk"][par]}
    z = torch.zeros(n, device=dev)
    return {"birth": torch.full((n,), float(tr.step), device=dev), "gen": z.clone(), "op_last": torch.full((n,), -1.0, device=dev),
            "ls_last": torch.full((n,), -99.0, device=dev), "risk": z.clone(), "parent_risk": z.clone()}


def ensure_aux(tr):
    if getattr(tr.cfg, "cags_mode", "none") == "none":
        return False
    n = tr.model.n
    if getattr(tr, "_aux", None) is None:
        tr._aux = {k: torch.zeros(0, device=tr.model.means.device) for k in
                   ("birth", "gen", "op_last", "ls_last", "risk", "parent_risk")}
        tr._aux = _aux_new(tr, n)
    elif tr._aux["birth"].numel() != n:          # out of sync (should not happen): reset
        tr._aux = _aux_new(tr, n)
    return True


def on_append(tr, n_new):
    par = tr.__dict__.pop("_par", None)
    if getattr(tr, "_aux", None) is None:
        return
    new = _aux_new(tr, n_new, par)
    tr._aux = {k: torch.cat([v, new[k]]) for k, v in tr._aux.items()}


def on_prune(tr, keep):
    if getattr(tr, "_aux", None) is not None and tr._aux["birth"].numel() == keep.numel():
        tr._aux = {k: v[keep] for k, v in tr._aux.items()}


# ---- GT labels (oracle / training targets only) -------------------------------------------------------------------
def _gt(tr):
    if not hasattr(tr, "_cags_gt"):
        P = torch.from_numpy(np.load(tr.cfg.gt_ref)).float()
        if len(P) > 50000:
            P = P[torch.randperm(len(P), generator=torch.Generator().manual_seed(0))[:50000]]
        P = P.to(tr.model.means.device)
        nrm = torch.zeros_like(P)                    # PCA normals from 16 nearest neighbours
        for i in range(0, len(P), 2048):
            d = torch.cdist(P[i:i + 2048], P)
            idx = torch.topk(d, 16, largest=False).indices
            Q = P[idx] - P[idx].mean(1, keepdim=True)
            C = Q.transpose(1, 2) @ Q
            _, vec = torch.linalg.eigh(C)
            nrm[i:i + 2048] = vec[..., 0]
        tr._cags_gt = (P, nrm)
    return tr._cags_gt


@torch.no_grad()
def gt_labels(tr):
    """d: centre distance to the STL reference; d_norm: d / std of the Gaussian along the reference normal.
    y_geo: 1 off-surface (d > tau and d_norm > 3), 0 on-surface (d < tau/2 or d_norm < 1), -1 ambiguous."""
    P, N = _gt(tr)
    m = tr.model
    mu = m.means.detach()
    d = torch.empty(m.n, device=mu.device)
    nn = torch.empty(m.n, dtype=torch.long, device=mu.device)
    for i in range(0, m.n, 2048):
        dd = torch.cdist(mu[i:i + 2048], P)
        v, j = dd.min(1)
        d[i:i + 2048], nn[i:i + 2048] = v, j
    n = N[nn]
    R = quat_to_rotmat(m.quats.detach() / m.quats.detach().norm(dim=-1, keepdim=True))
    v = torch.einsum("nji,nj->ni", R, n)                       # R^T n
    sig = (m.scales.detach() ** 2 * v ** 2).sum(-1).clamp_min(1e-12).sqrt()
    dn = d / sig
    tau = float(getattr(tr.cfg, "gt_tau", 0.1))
    y = torch.full((m.n,), -1.0, device=mu.device)
    y[(d < tau / 2) | (dn < 1)] = 0.0
    y[(d > tau) & (dn > 3)] = 1.0
    return d, dn, y


# ---- probe ----------------------------------------------------------------------------------------------------------
def _ro_appearance(tr, img, cam):
    """GC-ILA correction with the current coefficients, without updating them (read-only)."""
    if getattr(tr.cfg, "appearance", "none") != "gc_log" or cam.name not in getattr(tr, "gc", {}):
        return img
    a, b = tr.gc[cam.name]
    obj = (cam.image < 0.99).any(-1)
    return torch.where(obj[..., None], (a * img.clamp_min(1e-3).log() + b).exp(), img)


def _rank(x):
    n = x.numel()
    if n < 2:
        return torch.zeros_like(x)
    r = torch.empty(n, device=x.device)
    r[torch.argsort(x)] = torch.arange(n, device=x.device, dtype=torch.float32)
    return r / (n - 1)


def probe(tr, cams, with_labels=False):
    """Returns (F (N, len(FEATS)) normalized features, raw dict, labels dict or None)."""
    from frecon.data import photometric as ph
    cfg, m = tr.cfg, tr.model
    n, dev, V = m.n, m.means.device, len(cams)
    ext = float(tr.extent)
    mu = m.means.detach()
    W = torch.zeros(V, n, device=dev)
    Rabs = torch.zeros(V, n, device=dev)
    Rsgn = torch.zeros(V, n, device=dev)
    Lphi = torch.zeros(n, 3, device=dev)
    inb_c = torch.zeros(n, device=dev)
    front_c = torch.zeros(n, device=dev)
    ref_phi = torch.zeros(n, 3, device=dev)
    ref_cnt = torch.zeros(n, device=dev)
    ratio_l = []
    gdc_sum = torch.zeros(n, 3, device=dev)
    gdc_nrm = torch.zeros(n, device=dev)
    has_sh = "sh_rest" in m.params
    gsh_sum = torch.zeros(n, m.params["sh_rest"][0].numel() if has_sh else 1, device=dev)
    gsh_nrm = torch.zeros(n, device=dev)
    zmin = torch.full((n,), float("inf"), device=dev)
    have_phi = False
    thr = 0.5                                    # blending weight (pixels) for "the view sees the Gaussian"
    for j, cam in enumerate(cams):
        for p in m.params.values():
            p.grad = None
        Xc = mu @ cam.R.T + cam.t
        z = Xc[:, 2]
        cols = m.view_colors(cam)
        pr = torch.zeros(n, N_PROBE, device=dev, requires_grad=True)
        feats = torch.cat([cols, z[:, None], pr], -1)
        bg = torch.cat([tr.bg.to(dev).float(), torch.zeros(1 + N_PROBE, device=dev)])
        out = m.render(tr.rasterize, cam, bg, features=feats)
        img, Z, A = out["image"][..., :3], out["image"][..., 3], out["alpha"]
        aux = out["image"][..., 4:]
        tgt = cam.image.to(dev)
        pred = _ro_appearance(tr, img, cam)
        obj = (tgt < 0.99).any(-1)
        loss = (pred - tgt).abs().mean()
        with torch.no_grad():
            res = (pred - tgt).abs().mean(-1) * obj
            sgn = (tgt.clamp_min(1e-3).log() - pred.clamp_min(1e-3).log()).mean(-1) * obj
            clean = ph.__dict__.get("CLEAN", {}).get(cam.name)
            if clean is not None:
                clean = clean.to(dev)
                lphi = (tgt.clamp_min(1e-3).log() - clean.clamp_min(1e-3).log()) * obj[..., None]
                have_phi = True
            else:
                lphi = torch.zeros_like(tgt)
            cmap = torch.stack([torch.ones_like(res), res, sgn, lphi[..., 0], lphi[..., 1], lphi[..., 2]], -1)
        (loss + (cmap * aux).sum()).backward()
        with torch.no_grad():
            g = pr.grad
            w = g[:, 0].clamp_min(0)
            W[j], Rabs[j], Rsgn[j] = w, g[:, 1], g[:, 2]
            Lphi += g[:, 3:6]
            seen = w > thr
            gdc = m.params["color_logits"].grad
            if gdc is not None:
                gh = gdc / (w[:, None] + EPS) * seen[:, None]
                gdc_sum += gh
                gdc_nrm += gh.norm(dim=-1)
            if has_sh and m.params["sh_rest"].grad is not None:
                gs = m.params["sh_rest"].grad.reshape(n, -1) / (w[:, None] + EPS) * seen[:, None]
                gsh_sum += gs
                gsh_nrm += gs.norm(dim=-1)
            # centre projection: geometric visibility / "floating in front of the rendered surface"
            D = Z / A.clamp_min(1e-6)
            u = torch.floor(cam.fx * Xc[:, 0] / z.clamp_min(1e-6) + cam.cx).long()
            v = torch.floor(cam.fy * Xc[:, 1] / z.clamp_min(1e-6) + cam.cy).long()
            inb = (z > 1e-3) & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
            uu, vv = u.clamp(0, cam.W - 1), v.clamp(0, cam.H - 1)
            Dp, Ap = D[vv, uu], A[vv, uu]
            dlt = 0.02 * ext
            surf = inb & (Ap > 0.5)
            front = surf & (z < Dp - dlt)
            visg = inb & (~surf | (z <= Dp + dlt))     # not occluded by the rendered surface
            inb_c += inb.float()
            front_c += front.float()
            ref_phi += lphi[vv, uu] * visg[:, None]
            ref_cnt += visg.float()
            rr = torch.where(surf, z / Dp.clamp_min(1e-6), torch.full_like(z, float("nan")))
            ratio_l.append(rr)
            zmin = torch.minimum(zmin, torch.where(inb, z, torch.full_like(z, float("inf"))))
    for p in m.params.values():
        p.grad = None
    with torch.no_grad():
        wsum = W.sum(0)
        wh = W / (wsum + EPS)
        ent = -(wh * (wh + EPS).log()).sum(0) / math.log(max(V, 2))
        seenV = (W > thr).float()
        nvis = seenV.mean(0)
        maxs = wh.max(0).values
        res_abs = Rabs.sum(0) / (wsum + EPS)
        s_j = Rsgn / (W + EPS) * seenV
        cnt = seenV.sum(0).clamp_min(1)
        s_mean = s_j.sum(0) / cnt
        s_std = (((s_j - s_mean) ** 2) * seenV).sum(0).div(cnt).sqrt()
        conf_dc = torch.where(cnt > 1, 1 - gdc_sum.norm(dim=-1) / (gdc_nrm + EPS), torch.zeros_like(wsum))
        conf_sh = torch.where(cnt > 1, 1 - gsh_sum.norm(dim=-1) / (gsh_nrm + EPS), torch.zeros_like(wsum))
        front_frac = front_c / inb_c.clamp_min(1)
        ratio = torch.nanmedian(torch.stack(ratio_l), 0).values
        ratio = torch.nan_to_num(ratio, nan=1.0)
        op = m.opacities.detach().reshape(-1)
        ls = m.params["log_scales"].detach()
        ls_max, ls_min = ls.max(1).values, ls.min(1).values
        col = torch.sigmoid(m.params["color_logits"].detach())
        she = m.params["sh_rest"].detach().pow(2).flatten(1).sum(1) if has_sh else torch.zeros(n, device=dev)
        cen = mu.median(0).values
        dcen = (mu - cen).norm(dim=-1) / ext
        # k-NN context against a random subset of Gaussians
        g = torch.Generator(device="cpu").manual_seed(int(tr.step))
        sub = torch.randperm(n, generator=g)[:min(n, 20000)].to(dev)
        kd = torch.zeros(n, device=dev)
        ko, kf, ks, kc = (torch.zeros(n, device=dev) for _ in range(4))
        k = min(8, len(sub) - 1)
        if k >= 1:
            for i in range(0, n, 4096):
                dd = torch.cdist(mu[i:i + 4096], mu[sub])
                val, idx = torch.topk(dd, k + 1, largest=False)
                val, idx = val[:, 1:], sub[idx[:, 1:]]
                kd[i:i + 4096] = val.mean(1) / ext
                ko[i:i + 4096] = op[idx].mean(1)
                kf[i:i + 4096] = front_frac[idx].mean(1)
                ks[i:i + 4096] = maxs[idx].mean(1)
                kc[i:i + 4096] = conf_dc[idx].mean(1)
        a = tr._aux
        age = (tr.step - a["birth"]) / max(cfg.iterations, 1)
        d_op = torch.where(a["op_last"] < 0, torch.zeros_like(op), op - a["op_last"])
        d_ls = torch.where(a["ls_last"] < -50, torch.zeros_like(ls_max), ls_max - a["ls_last"])
        a["op_last"], a["ls_last"] = op.clone(), ls_max.clone()
        gacc = tr.grad_accum / tr.grad_denom.clamp_min(1) if tr.grad_accum.numel() == n else torch.zeros(n, device=dev)
        raw = {"opacity": op, "ls_max": ls_max, "ls_min": ls_min, "aniso": ls_max - ls_min, "col_r": col[:, 0],
               "col_g": col[:, 1], "col_b": col[:, 2], "sh_energy": she, "dist_center": dcen, "depth_ratio": ratio,
               "n_vis": nvis, "entropy": ent, "max_share": maxs, "front_frac": front_frac,
               "log_contrib": (wsum + 1).log(), "knn_dist": kd, "knn_opacity": ko, "knn_front": kf, "knn_share": ks,
               "res_abs": res_abs, "res_sgn_std": s_std, "conf_dc": conf_dc.clamp(0, 1), "conf_sh": conf_sh.clamp(0, 1),
               "knn_conf": kc.clamp(0, 1), "age": age.clamp(0, 1), "gen": a["gen"], "d_opacity": d_op, "d_ls": d_ls,
               "grad_acc": gacc, "parent_risk": a["parent_risk"],
               "t_norm": torch.full((n,), tr.step / max(cfg.iterations, 1), device=dev)}
        F = torch.stack([raw[nm] if bnd else _rank(raw[nm]) for nm, _, bnd in FEATS], 1)
        F = torch.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0)
        lab = None
        if with_labels and getattr(cfg, "gt_ref", ""):
            d, dn, y = gt_labels(tr)
            if have_phi:
                obs = Lphi / (wsum[:, None] + EPS)
                ref = ref_phi / ref_cnt.clamp_min(1)[:, None]
                delta = (obs - ref).norm(dim=-1) * (wsum > thr)
            else:
                delta = torch.zeros(n, device=dev)
            on = (y == 0) & (wsum > thr)
            tau_d = float(torch.quantile(delta[on], 0.9)) if have_phi and int(on.sum()) > 10 else float("inf")
            lab = {"d": d, "d_norm": dn, "y_geo": y, "delta": delta, "tau_delta": tau_d, "wsum": wsum,
                   "y_cmp": (delta > tau_d).float(), "y_joint": ((y == 1) & (delta > tau_d)).float()}
    return F, raw, lab


# ---- detector -----------------------------------------------------------------------------------------------------
class Detector(torch.nn.Module):
    def __init__(self, idx, hidden=128, depth=3):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(idx, dtype=torch.long))
        layers, d = [], len(idx)
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU()]
            d = hidden
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(hidden, 2)        # logit p_geo, delta_hat (in units of tau_delta)

    def forward(self, F):
        h = self.body(F[:, self.idx])
        o = self.head(h)
        return o[:, 0], o[:, 1]


def joint_risk(logit, dhat, T=0.25, use_cmp=True):
    p = torch.sigmoid(logit)
    return p * torch.sigmoid((dhat - 1.0) / T) if use_cmp else p


def load_detector(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    det = Detector(ck["idx"], ck.get("hidden", 128), ck.get("depth", 3)).to(dev)
    det.load_state_dict(ck["state"])
    det.eval()
    det.use_cmp = ck.get("use_cmp", True)
    return det


def rule_score(F):
    """GT-free hand-made baseline: mean of the ranks of max_share, front_frac, conf_dc and res_abs."""
    cols = [FEAT_NAMES.index(c) for c in ("max_share", "front_frac", "conf_dc", "res_abs")]
    return torch.stack([_rank(F[:, c]) for c in cols], 1).mean(1)


# ---- hooks called from Trainer --------------------------------------------------------------------------------------
def _strength(tr):
    cfg, r = tr.cfg, tr._aux["risk"]
    ts = float(getattr(cfg, "cags_tau_s", 0.5))
    m = ((r - ts) / (1 - ts)).clamp(0, 1)
    b = float(getattr(cfg, "cags_budget", 1.0))
    if b < 1.0 and r.numel():
        k = max(int(b * r.numel()), 1)
        thr = torch.topk(r, k).values.min()
        m = m * (r >= thr)
    return m


def _acts(cfg):
    a = getattr(cfg, "cags_actions", "full")
    return {"gate", "att", "mod"} if a == "full" else set(a.split("+"))


def _active(tr):
    cfg = tr.cfg
    return (getattr(cfg, "cags_mode", "none") not in ("none", "log") and tr.step >= getattr(cfg, "cags_from", 0)
            and getattr(tr, "_aux", None) is not None and tr._aux["risk"].numel() == tr.model.n)


def gate(tr):
    if not _active(tr) or "gate" not in _acts(tr.cfg):
        return None
    return tr._aux["risk"] > float(getattr(tr.cfg, "cags_tau_gate", 0.5))


def pre_step(tr):
    """Before the optimizer step: rows to modulate and their current values."""
    if not _active(tr) or "mod" not in _acts(tr.cfg):
        return None
    m = _strength(tr)
    sel = m > 0
    if not bool(sel.any()):
        return None
    P = tr.model.params
    keep = {k: P[k].detach()[sel].clone() for k in ("color_logits", "sh_rest", "log_scales") if k in P}
    return sel, m[sel], keep


def post_step(tr, pre):
    if not _active(tr):
        return
    P = tr.model.params
    with torch.no_grad():
        if pre is not None:
            sel, ms, keep = pre
            for k, old in keep.items():
                new = P[k].detach()[sel]
                f = (1 - ms).reshape(-1, *([1] * (old.dim() - 1)))
                P[k][sel] = old + f * (new - old)
        if "att" in _acts(tr.cfg):
            m = _strength(tr)
            sel = m > 0
            if bool(sel.any()):
                gam = float(getattr(tr.cfg, "cags_gamma", 2.0))
                cap = inverse_sigmoid(((1 - m[sel]) ** gam).clamp(1e-4, 1.0))
                lg = P["opacity_logits"]
                lg[sel] = torch.minimum(lg[sel], cap.reshape(lg[sel].shape))


def tick(tr):
    """After a training step (and densification): probe when due, log features / labels, update the risk EMA."""
    cfg = tr.cfg
    mode = getattr(cfg, "cags_mode", "none")
    if mode == "none" or not ensure_aux(tr):
        return
    s = tr.step
    due_log = s in set(int(x) for x in (getattr(cfg, "cags_log_steps", ()) or ()))
    t0, K = int(getattr(cfg, "cags_from", 0)), int(getattr(cfg, "cags_every", 100))
    first = not getattr(tr, "_cags_init", False)
    due_act = mode != "log" and s >= t0 and (first or (s - t0) % K == 0)
    if not (due_log or due_act):
        return
    cams = getattr(tr, "_cams", None)
    if not cams:
        return
    need_lab = due_log or mode.startswith("oracle")
    F, raw, lab = probe(tr, cams, with_labels=need_lab)
    if due_log and lab is not None:
        _log(tr, F, lab)
        if mode != "log" and getattr(tr, "_cags_init", False):   # diagnostics against GT (logging only)
            act, yg = _strength(tr) > 0, lab["y_geo"]
            tr.cags_hist.append({"step": s, "diag": 1, "n": tr.model.n, "n_off": int((yg == 1).sum()),
                                 "act_off": int((act & (yg == 1)).sum()), "act_on": int((act & (yg == 0)).sum()),
                                 "act_joint": int((act & (lab["y_joint"] > 0)).sum()), "n_joint": int(lab["y_joint"].sum())})
    if not due_act:
        return
    with torch.no_grad():
        if mode == "oracle_geo":
            q = (lab["y_geo"] == 1).float()
        elif mode == "oracle_cmp":
            q = lab["y_joint"]
        elif mode == "rule":
            q = rule_score(F)
        elif mode == "det":
            if not hasattr(tr, "_det"):
                tr._det = load_detector(cfg.cags_model, F.device)
            lg, dh = tr._det(F)
            q = joint_risk(lg, dh, use_cmp=tr._det.use_cmp)
        else:
            raise ValueError(mode)
        b = float(getattr(cfg, "cags_ema", 0.5))
        r = tr._aux["risk"]
        tr._aux["risk"] = q.float() if first else b * r + (1 - b) * q.float()
        tr._cags_init = True
        row = {"step": s, "n": tr.model.n, "risk_mean": float(tr._aux["risk"].mean()),
               "n_active": int((_strength(tr) > 0).sum()),
               "n_gate": int((tr._aux["risk"] > float(getattr(cfg, "cags_tau_gate", 0.5))).sum())}
        tr.__dict__.setdefault("cags_hist", []).append(row)


def _log(tr, F, lab):
    """Stratified subsample of (features, labels) rows kept on the CPU (float16)."""
    n = F.shape[0]
    g = torch.Generator().manual_seed(int(tr.step) * 31 + 7)
    y = lab["y_geo"].cpu()
    pos = torch.nonzero(y == 1).flatten()
    oth = torch.nonzero(y != 1).flatten()
    cap = int(getattr(tr.cfg, "cags_log_max", 20000))
    pos = pos[torch.randperm(len(pos), generator=g)[:cap]]
    oth = oth[torch.randperm(len(oth), generator=g)[:cap]]
    idx = torch.cat([pos, oth]).sort().values
    wpos = max(int((y == 1).sum()), 1) / max(len(pos), 1)
    woth = max(int((y != 1).sum()), 1) / max(len(oth), 1)
    sw = torch.where(y[idx] == 1, torch.tensor(wpos), torch.tensor(woth))   # inverse sampling rate
    rows = {"F": F[idx.to(F.device)].half().cpu(), "step": torch.full((len(idx),), tr.step, dtype=torch.int32),
            "sw": sw.float()}
    for k in ("d", "d_norm", "y_geo", "delta", "wsum", "y_cmp", "y_joint"):
        rows[k] = lab[k][idx.to(F.device)].float().cpu()
    rows["tau_delta"] = torch.full((len(idx),), lab["tau_delta"])
    tr.__dict__.setdefault("cags_rows", []).append(rows)
    tr.__dict__.setdefault("cags_hist", []).append(
        {"step": tr.step, "n": n, "n_off": int((y == 1).sum()), "n_amb": int((y == -1).sum()),
         "tau_delta": lab["tau_delta"], "frac_joint": float(lab["y_joint"].mean())})


def write_log(tr, path):
    rows = getattr(tr, "cags_rows", None)
    if rows:
        out = {k: torch.cat([r[k] for r in rows]).numpy() for k in rows[0]}
        np.savez_compressed(path.parent / "cags_log.npz", feat_names=np.array(FEAT_NAMES), **out)
    hist = getattr(tr, "cags_hist", None)
    if hist:
        import csv
        keys = sorted({k for r in hist for k in r})
        with open(path.parent / "cags_hist.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(hist)
