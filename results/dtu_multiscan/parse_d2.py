import re, sys, glob, os
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
R = sys.argv[1]
rows, meas = [], []
for f in sorted(glob.glob(f'{R}/runs/d2v21_dtu_scan*_v8/report.md')):
    scan = int(re.search(r'scan(\d+)', f).group(1))
    txt = open(f, encoding='utf-8').read()
    for step in ('0', '200', '1000'):
        sec = re.search(rf'## Gains at step {step}[^\n]*\n\n(\|.*?)\n\n', txt, re.S).group(1).splitlines()
        hdr = [h.strip() for h in sec[0].strip('|').split('|')]
        for line in sec[2:]:
            cells = [c.strip() for c in line.strip('|').split('|')]
            sig = cells[0].split()[0]
            for h, c in zip(hdr[1:], cells[1:]):
                m = re.match(r'([+-]?[\d.]+) ± ([\d.]+)\s*(\*\*([EO])\*\*)?', c)
                if m:
                    rows.append(dict(scan=scan, step=int(step), signal=sig, metric=h, mean=float(m[1]), std=float(m[2]), cls=m[4] or '·'))
    for line in re.search(r'## Measurability.*?\n\n(\|.*?)\n\n', txt, re.S).group(1).splitlines()[2:]:
        c = [x.strip() for x in line.strip('|').split('|')]
        meas.append(dict(scan=scan, signal=c[0], snr1000=c[2], measurable=c[3]))
    dec = re.findall(r'- \*\*(\w+)\*\*: (.*)', txt)
    for s, d in dec: meas.append(dict(scan=scan, signal=s, decision=d))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
t = df[df.step == 1000]
short = {'Gain_LR image (region mask)': 'LR_img', 'Gain_LR 3D displacement': 'LR_3D', 'Gain_Dir held-out 1−SSIM': 'heldout', 'Gain_Dir local accuracy': 'acc', 'Gain_Dir local completeness': 'comp'}
t = t.assign(metric=t.metric.map(short))
piv = t.pivot_table(index=['signal', 'metric'], columns='scan', values='mean')
cls = t.pivot_table(index=['signal', 'metric'], columns='scan', values='cls', aggfunc='first')
print(piv.round(3)); print(cls)
out = []
for (s, m), v in piv.iterrows():
    v = v.dropna().values
    out.append(dict(signal=s, metric=m, n=len(v), mean=v.mean(), n_pos=int((v > 0).sum()), p=wilcoxon(v).pvalue))
o = pd.DataFrame(out); print(o.round(4).to_string(index=False))
prim = o[o.metric.isin(['comp', 'LR_3D']) & o.signal.isin(['gradmag', 'visibility'])].sort_values('p').reset_index(drop=True)
k = len(prim); adj = []; run = 0
for i, p in enumerate(prim.p): run = max(run, min(1, (k - i) * p)); adj.append(run)
prim['p_holm'] = adj; print('\nPRIMARY\n', prim.round(4).to_string(index=False))
mz = pd.DataFrame(meas); print(mz.dropna(subset=['measurable']).pivot(index='signal', columns='scan', values='measurable'))
print(mz.dropna(subset=['decision']).pivot(index='signal', columns='scan', values='decision').to_string())
t0 = df[df.step == 0].assign(metric=lambda x: x.metric.map(short))
print('\nSTEP 0 heldout\n', t0[t0.metric=='heldout'].pivot_table(index='signal', columns='scan', values='mean').round(2))
df.to_csv(f'{R}/d2_all_scans_long.csv', index=False)
