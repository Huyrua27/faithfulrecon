import glob, re, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
R = sys.argv[1]; rows = []; info = []
for f in sorted(glob.glob(f'{R}/runs/d1/dtu_scan*/v8/seed_*/scores.npz')):
    scan = int(re.search(r'scan(\d+)', f).group(1)); seed = int(re.search(r'seed_(\d+)', f).group(1))
    sc = np.load(f, allow_pickle=True); lab = np.load(f.replace('scores.npz', 'geometric_labels.npz'))
    d, fi = lab['distance'], lab['f_i'].astype(bool)
    info.append(dict(scan=scan, seed=seed, N=len(d), far01=(d > 0.1).mean(), far003=(d > 0.03).mean(), f_i=fi.mean(), signals=','.join(k for k in sc.files if k != 'meta')))
    for k in sc.files:
        if k == 'meta': continue
        x = sc[k]
        for ln, y in (('f_i', fi), ('d>0.03', d > 0.03), ('d>0.1', d > 0.1)):
            rows.append(dict(scan=scan, seed=seed, signal=k, label=ln, auroc=roc_auc_score(y, x)))
        rows.append(dict(scan=scan, seed=seed, signal=k, label='spearman_d', auroc=spearmanr(x, d).statistic))
pd.set_option('display.width', 250)
I = pd.DataFrame(info); print(I.groupby('scan').agg(N=('N', 'mean'), far01=('far01', 'mean'), far003=('far003', 'mean'), f_i=('f_i', 'mean'), signals=('signals', 'first')).round(3).to_string())
A = pd.DataFrame(rows); A.to_csv(f'{R}/d1_all_scans.csv', index=False)
ps = A.groupby(['label', 'signal', 'scan']).auroc.mean().unstack('scan')
for ln in ('d>0.1', 'd>0.03', 'f_i', 'spearman_d'):
    print('\n==', ln); t = ps.loc[ln]; t['mean'] = t.mean(1); print(t.round(3).to_string())
