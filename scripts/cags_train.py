"""Train / evaluate the CAGS detector on logged (features, labels) rows, leave-scan-out.

python scripts/cags_train.py --logs "runs/v5_*_std_van_dtu_scan*_v8/seed_*/cags_log.npz" --out results/v5/det \
    --folds 24,63,97 37,65,105 ... --sets A AG AGP AGPD --train_levels none pa pb pc --tag r0

For every fold f and feature set S a detector is trained on the scans outside f and evaluated on the scans of f.
Saved: <out>/<tag>_<S>_fold<f>.pt (used by cags_mode=det) and <out>/<tag>_metrics.csv. Labels: y_geo (off-surface,
-1 ambiguous rows are excluded from the classification loss), delta / tau_delta (compensation, regression head).
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frecon.cags import FEATS, GROUPS, Detector, joint_risk  # noqa: E402

RUN = re.compile(r"v5_(none|pac|pa|pb|pc)_std_(.+)_dtu_scan(\d+)_v8")


def load(pattern):
    parts = []
    for f in sorted(glob.glob(pattern)):
        m = RUN.search(f)
        if not m:
            continue
        z = np.load(f)
        n = len(z["y_geo"])
        d = {k: z[k] for k in ("F", "y_geo", "delta", "tau_delta", "wsum", "y_joint", "sw", "step", "d", "d_norm")}
        d.update(level=np.full(n, m[1]), tag=np.full(n, m[2]), scan=np.full(n, int(m[3])),
                 seed=np.full(n, int(Path(f).parent.name.split("_")[1])))
        parts.append(d)
    if not parts:
        raise SystemExit(f"no logs match {pattern}")
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # LABEL_V2: labels recomputed from the logged distances (d > 0.1 is off-surface whatever d_norm);
    # sw (inverse sampling rate of the logged strata) stays valid
    y = np.full(len(D['d']), -1.0, np.float32)
    y[(D['d'] < 0.05) | ((D['d_norm'] < 1) & (D['d'] <= 0.1))] = 0.0
    y[D['d'] > 0.1] = 1.0
    D['y_geo'] = y
    D['y_joint'] = ((y == 1) & (D['delta'] > D['tau_delta'])).astype(np.float32)
    return D


def feat_idx(s):
    return sorted(i for g in s for i in GROUPS[g])


def targets(D, rows):
    tau = D["tau_delta"][rows]
    t = np.where(np.isfinite(tau) & (tau > 0), D["delta"][rows] / np.where(np.isfinite(tau) & (tau > 0), tau, 1), 0.0)
    return np.clip(t, 0, 5).astype(np.float32)


def train_one(D, rows, idx, dev, epochs=8, bs=16384, lr=2e-3, max_rows=4_000_000, seed=0):
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        rows = rng.choice(rows, max_rows, replace=False)
    F = torch.from_numpy(D["F"][rows]).to(dev)
    y = torch.from_numpy(D["y_geo"][rows]).float().to(dev)
    t = torch.from_numpy(targets(D, rows)).to(dev)
    w = torch.from_numpy((D["sw"][rows] * np.log1p(D["wsum"][rows])).astype(np.float32)).to(dev)
    valid = y >= 0
    pos, neg = (w * (y == 1)).sum(), (w * (y == 0)).sum()
    wc = torch.where(y == 1, w / pos.clamp_min(1e-9), w / neg.clamp_min(1e-9)) * valid * 0.5 * len(y)
    wr = w / w.mean().clamp_min(1e-9)
    torch.manual_seed(seed)
    det = Detector(idx).to(dev)
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(y) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    det.train()
    for _ in range(steps):
        b = torch.randint(0, len(y), (bs,), device=dev)
        lg, dh = det(F[b].float())
        p = torch.sigmoid(lg)
        yb = y[b].clamp_min(0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, yb, reduction="none")
        pt = torch.where(yb > 0, p, 1 - p)
        cls = (wc[b] * (1 - pt) ** 2 * bce).mean()
        reg = (wr[b] * torch.nn.functional.huber_loss(dh, t[b], reduction="none", delta=1.0)).mean()
        loss = cls + 0.5 * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    det.eval()
    return det


@torch.no_grad()
def predict(det, F, dev, bs=262144, use_cmp=True):
    lg, dh = [], []
    for i in range(0, len(F), bs):
        a, b = det(torch.from_numpy(F[i:i + bs]).to(dev).float())
        lg.append(a.cpu())
        dh.append(b.cpu())
    lg, dh = torch.cat(lg), torch.cat(dh)
    return torch.sigmoid(lg).numpy(), dh.numpy(), joint_risk(lg, dh, use_cmp=use_cmp).numpy()


def calibrate(D, rows, det, dev, prec=0.8, use_cmp=True, n=1_000_000, seed=0, rate=None):
    """Lowest threshold tau on the risk q at which the flagged training Gaussians are off-surface (y_geo = 1) with
    weighted precision >= prec, i.e. the largest recall at that precision. Falls back to the max-F1 threshold.
    With rate: the threshold that flags that (weighted) fraction of the training Gaussians instead."""
    rng = np.random.default_rng(seed)
    rows = rows[D["y_geo"][rows] >= 0]
    if len(rows) > n:
        rows = rng.choice(rows, n, replace=False)
    _, _, q = predict(det, D["F"][rows], dev, use_cmp=use_cmp)
    y = D["y_geo"][rows] == 1
    w = D["sw"][rows].astype(np.float64)
    o = np.argsort(-q)
    tp, fl = np.cumsum((w * y)[o]), np.cumsum(w[o])
    pr, rc = tp / fl, tp / max(tp[-1], 1e-12)
    ok = np.nonzero(pr >= prec)[0] if rate is None else np.array([min(np.searchsorted(fl / fl[-1], rate), len(q) - 1)])
    k = ok.max() if len(ok) else int(np.argmax(2 * pr * rc / np.maximum(pr + rc, 1e-12)))
    tau = float(max(q[o][k], 1e-6))
    info = dict(tau=tau, prec=float(pr[k]), recall_off=float(rc[k]), flag_rate=float(fl[k] / fl[-1]), fallback=not len(ok))
    return tau, info


def rule(F):
    names = [f[0] for f in FEATS]
    cols = [names.index(c) for c in ("max_share", "front_frac", "conf_dc", "res_abs")]
    r = np.stack([pd.Series(F[:, c].astype(np.float32)).rank(pct=True).to_numpy() for c in cols], 1)
    return r.mean(1)


def scores(D, rows, s_geo, s_joint, s_delta, name):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = D["y_geo"][rows]
    v = y >= 0
    w = D["sw"][rows] * D["wsum"][rows]
    out = {"model": name, "n": int(v.sum()), "pos_rate": float((y[v] == 1).mean()) if v.any() else np.nan}
    if v.any() and 0 < (y[v] == 1).mean() < 1:
        out["auroc_geo"] = roc_auc_score(y[v] == 1, s_geo[v], sample_weight=D["sw"][rows][v])
        out["ap_geo"] = average_precision_score(y[v] == 1, s_geo[v], sample_weight=D["sw"][rows][v])
        out["ap_geo_w"] = average_precision_score(y[v] == 1, s_geo[v], sample_weight=w[v])
    yj = D["y_joint"][rows] > 0.5
    if 0 < yj.mean() < 1:
        out["ap_joint"] = average_precision_score(yj, s_joint, sample_weight=D["sw"][rows])
        out["ap_joint_w"] = average_precision_score(yj, s_joint, sample_weight=w)
        out["joint_rate"] = float(yj.mean())
    pert = D["level"][rows] != "none"
    if s_delta is not None and pert.sum() > 100:
        out["spearman_delta"] = spearmanr(s_delta[pert], D["delta"][rows][pert]).correlation
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", nargs="+", required=True, help="comma-separated scans per fold")
    ap.add_argument("--sets", nargs="+", default=["A", "AG", "AGP", "AGPD"])
    ap.add_argument("--train_levels", nargs="+", default=["none", "pa", "pb", "pc", "pac"])
    ap.add_argument("--test_levels", nargs="+", default=["none", "pa", "pb", "pc", "pac"])
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--cal_prec", type=float, default=0.8, help="precision (off-surface) of the calibration threshold")
    ap.add_argument("--cal_rate", type=float, default=None, help="calibrate to this flag rate instead of a precision")
    ap.add_argument("--calibrate_only", action="store_true", help="recalibrate existing <tag>_<set>_fold<f>.pt in place")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts = [load(p) for p in a.logs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f"{len(D['y_geo'])} rows, scans {sorted(set(D['scan']))}, levels {sorted(set(D['level']))}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = [[int(s) for s in f.split(",")] for f in a.folds]
    res = []
    for fi, fs in enumerate(folds):
        tr_rows = np.nonzero(~np.isin(D["scan"], fs) & np.isin(D["level"], a.train_levels))[0]
        te_rows = np.nonzero(np.isin(D["scan"], fs) & np.isin(D["level"], a.test_levels))[0]
        if len(tr_rows) == 0 or len(te_rows) == 0:
            print(f"fold {fi}: no rows, skipped")
            continue
        for s in a.sets:
            f = out / f"{a.tag}_{s}_fold{fi}.pt"
            if a.calibrate_only:
                ck = torch.load(f, map_location=dev, weights_only=False)
                det = Detector(ck["idx"], ck.get("hidden", 128), ck.get("depth", 3)).to(dev)
                det.load_state_dict(ck["state"])
                det.eval()
            else:
                idx = feat_idx(s)
                det = train_one(D, tr_rows, idx, dev, epochs=a.epochs, seed=fi)
                ck = {"idx": idx, "state": det.state_dict(), "hidden": 128, "depth": 3, "use_cmp": True,
                      "set": s, "fold": fs, "train_levels": a.train_levels}
            tau, info = calibrate(D, tr_rows, det, dev, a.cal_prec, ck.get("use_cmp", True), seed=fi, rate=a.cal_rate)
            ck.update(tau=tau, cal=info)
            torch.save(ck, f)
            print(f"fold {fi} set {s}: tau {tau:.4f}, precision {info['prec']:.3f}, recall_off {info['recall_off']:.3f}, "
                  f"flag rate {info['flag_rate']:.4f}" + (" (max-F1 fallback)" if info["fallback"] else ""), flush=True)
            if a.calibrate_only:
                continue
            for lv in a.test_levels + ["all"]:
                rr = te_rows if lv == "all" else te_rows[D["level"][te_rows] == lv]
                if len(rr) == 0:
                    continue
                pg, dh, q = predict(det, D["F"][rr], dev)
                res.append(dict(scores(D, rr, pg, q, dh, s), fold=fi, level=lv, tag=a.tag))
            print(f"fold {fi} set {s}: done", flush=True)
        if a.baselines:
            names = [f[0] for f in FEATS]
            for lv in a.test_levels + ["all"]:
                rr = te_rows if lv == "all" else te_rows[D["level"][te_rows] == lv]
                if len(rr) == 0:
                    continue
                F = D["F"][rr].astype(np.float32)
                for bn, sc in (("rule", rule(F)), ("opacity", F[:, names.index("opacity")]),
                               ("max_share", F[:, names.index("max_share")]), ("front_frac", F[:, names.index("front_frac")])):
                    res.append(dict(scores(D, rr, sc, sc, None, bn), fold=fi, level=lv, tag=a.tag))
    df = pd.DataFrame(res)
    if df.empty:          # --calibrate_only: metrics unchanged
        return
    df.to_csv(out / f"{a.tag}_metrics.csv", index=False)
    cols = [c for c in ("auroc_geo", "ap_geo", "ap_geo_w", "ap_joint", "ap_joint_w", "spearman_delta") if c in df]
    print(df[df.level == "all"].groupby("model")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
