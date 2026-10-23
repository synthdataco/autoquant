"""Offline: forecast next-H-minute deseasonalised realised variance from EWMAs of deseasonalised r^2 at several
half-lives. Compare the served 2-EWMA rule with a log-linear combination fitted on train (< 09-08), on test.
uv run agent/analysis/har.py 60|1440"""
import sys, os, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, '/workspace/agent/analysis')
from season import load
os.environ['SEASON_FILE'] = 'season_pre0908_hl21_l1_ev_moh.json'
spec = importlib.util.spec_from_file_location('m', '/workspace/agent/modeling.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
H = int(sys.argv[1]); step = 1 if H == 60 else 5
HLS = [5, 15, 30, 60, 120, 240, 720, 2880]
assets = sys.argv[2].split(',') if len(sys.argv) > 2 else ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE']
rows = []
for a in assets:
    c = load(a)['2026-05-20':'2026-10-11']
    idx = c.index
    pm = m._week_minute_profile(a, idx, step) * m._event_mult(a, idx, step)
    lp = np.log(c.ffill())
    r = (lp - lp.shift(step)).to_numpy()
    cum = np.concatenate([[0], np.cumsum(pm)])
    s = np.full(len(c), np.nan); s[step:] = cum[step + 1:] - cum[1:-step]
    u2 = pd.Series(r**2, idx); ss = pd.Series(s, idx)
    ok = u2.notna() & ss.notna()
    feats = {}
    for hl in HLS:
        al = 1 - 0.5 ** (1 / hl)
        feats[hl] = u2.where(ok, 0).ewm(alpha=al, adjust=False).mean() / ss.where(ok, 0).ewm(alpha=al, adjust=False).mean()
    # target: next H minutes of 1-min r^2 over the profile sum (the 1-min variance actually scored), starting 2 min later
    r1 = (lp - lp.shift(1)).to_numpy() ** 2
    pm1 = m._week_minute_profile(a, idx, 1) * m._event_mult(a, idx, 1)
    v = pd.Series(np.where(np.isfinite(r1), r1, 0), idx); w = pd.Series(np.where(np.isfinite(r1), pm1, 0), idx)
    fy = v[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-2); fs = w[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-2)
    y = fy / fs
    samp = idx[(idx >= '2026-06-01') & (idx.minute % 7 == 0)]
    d = pd.DataFrame({f'f{hl}': feats[hl].reindex(samp) for hl in HLS})
    d['y'] = y.reindex(samp).values; d['asset'] = a; d['t'] = samp
    rows.append(d)
d = pd.concat(rows).replace([np.inf, -np.inf], np.nan).dropna()
d = d[(d.y > 0) & (d[[f'f{h}' for h in HLS]] > 0).all(axis=1)]
tr, te = d[d.t < '2026-09-08'], d[d.t >= '2026-09-08']
def qlike(f, y): x = y / f; return float(np.mean(x - np.log(x) - 1))
k = np.arange(H) + 2
hs, tau = (30, 60) if H == 60 else (120, 240)
def served(df):
    fs_, fl_ = df[f'f{hs}'], df['f2880']
    return fl_ + (fs_ - fl_) * np.mean(np.exp(-k / tau))
X = lambda df: np.column_stack([np.ones(len(df))] + [np.log(df[f'f{h}']) for h in HLS])
beta = np.linalg.lstsq(X(tr), np.log(tr.y), rcond=None)[0]
# log-linear gives the median; scale to the mean on train
c0 = np.log(np.mean(tr.y / np.exp(X(tr) @ beta)))
har = lambda df: np.exp(X(df) @ beta + c0)
lin = np.linalg.lstsq(np.column_stack([df_ for df_ in [tr[f'f{h}'] for h in HLS]]), tr.y, rcond=None)[0]
linf = lambda df: np.column_stack([df[f'f{h}'] for h in HLS]) @ lin
print('beta', np.round(beta, 3), 'linear w', np.round(lin, 3))
for name, df in [('train', tr), ('test', te)]:
    out = {'served': served(df), 'har_log': har(df), 'lin': np.maximum(linf(df), 1e-12)}
    for hl in HLS: out[f'f{hl}'] = df[f'f{hl}']
    print(name, len(df), ' '.join(f'{k_}:{qlike(f, df.y):.4f}' for k_, f in out.items()))
    print('  mean log(y/f)^2:', ' '.join(f'{k_}:{np.mean(np.log(df.y / f) ** 2):.4f}' for k_, f in list(out.items())[:3]))
