"""Exploration (not CRPS): QLIKE-optimal simplex weights on 3 EWMAs of deseasonalised variance,
per lead window, pooled over a group of assets. Data before --cutoff only. Prints JSON."""
import sys, json, argparse, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
ap = argparse.ArgumentParser()
ap.add_argument('--assets', required=True); ap.add_argument('--kind', default='v5')
ap.add_argument('--hls', required=True, help='comma list of 3 half-lives (min)')
ap.add_argument('--leads', required=True, help='e.g. 0-60,60-240,240-720,720-1440')
ap.add_argument('--cutoff', default='2026-08-01'); ap.add_argument('--prof', default='agent/research/profiles_2026-08-01.npz')
ap.add_argument('--from', dest='frm', default='2025-11-01'); ap.add_argument('--clipmed', type=float, default=50)
ap.add_argument('--measure', default='sq', choices=['sq', 'abs'])
a = ap.parse_args()
k = 5 if a.kind == 'v5' else 1
HLS = [float(x) for x in a.hls.split(',')]
LEADS = [tuple(int(v) for v in s.split('-')) for s in a.leads.split(',')]
prof = np.load(a.prof)
T = {l: [] for l in LEADS}; X = {l: [] for l in LEADS}
for asset in a.assets.split(','):
    df = load_all(asset, end=(pd.Timestamp(a.cutoff) - pd.Timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M'))
    df = df[df.index >= pd.Timestamp(a.frm, tz='UTC')]
    lp = np.log(df['close'].to_numpy())
    ny = df.index.tz_convert('America/New_York')
    mow = (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()
    r = np.full(len(lp), np.nan); r[:-k] = lp[k:] - lp[:-k]
    pk = ('a' if a.measure == 'abs' else 'v') + a.kind[1]
    z = r ** 2 / (k * np.maximum(prof[f'{asset}_{pk}'][mow], 0.03))
    if a.measure == 'abs':
        z = np.sqrt(z)
    slow = pd.Series(z).ewm(halflife=2880, ignore_na=True, min_periods=60).mean().to_numpy()
    zc = np.minimum(z, (np.sqrt(a.clipmed) if a.measure == 'abs' else a.clipmed) * slow)
    feats = [pd.Series(zc).ewm(halflife=h, ignore_na=True, min_periods=60).mean().to_numpy() for h in HLS]
    step = 60 if k == 5 else 20
    origins = np.arange(14 * 1440, len(lp) - LEADS[-1][1] - k, step)
    cs = np.concatenate([[0], np.nancumsum(z)]); cn = np.concatenate([[0], np.cumsum(np.isfinite(z))])
    for (l0, l1) in LEADS:
        tgt = (cs[origins + l1] - cs[origins + l0]) / np.maximum(cn[origins + l1] - cn[origins + l0], 1)
        F = np.stack([f[origins - k] for f in feats], 1)
        ok = np.isfinite(tgt) & (tgt > 0) & np.all(np.isfinite(F) & (F > 0), 1)
        T[(l0, l1)].append(tgt[ok]); X[(l0, l1)].append(F[ok])
out = {}
grid = [(w1, w2, round(1 - w1 - w2, 2)) for w1 in np.arange(0, 1.001, 0.05) for w2 in np.arange(0, 1.001 - w1, 0.05)]
for l in LEADS:
    t = np.concatenate(T[l]); F = np.concatenate(X[l])
    best = None
    for w in grid:
        f = F @ np.array(w)
        f = np.maximum(f, 1e-30)
        # optimal scale for QLIKE is mean(t/f)
        sc = np.mean(t / f)
        q = np.mean(t / (sc * f) - np.log(t / (sc * f)) - 1)
        if best is None or q < best[0]:
            best = (q, w, sc)
    out[f'{l[0]}-{l[1]}'] = {'w': [round(float(x), 2) for x in best[1]], 'scale': round(float(best[2]), 3), 'q': round(float(best[0]), 4), 'n': int(len(t))}
print(json.dumps(out))
