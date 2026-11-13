"""Exploration (not CRPS): which EWMA mix best forecasts deseasonalised variance at each lead?
QLIKE loss of convex 2-EWMA combos per lead window; hourly origins; profiles from a file."""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
W = 7 * 1440
prof = np.load('agent/research/profiles_2026-10-01.npz')
assets = sys.argv[1].split(',') if len(sys.argv) > 1 else ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE']
kind = sys.argv[2] if len(sys.argv) > 2 else 'v5'
k = 5 if kind == 'v5' else 1
HLS = [10, 30, 60, 120, 240, 480, 960, 1440, 2880, 5760] if k == 1 else [30, 60, 120, 240, 480, 960, 1440, 2880, 5760, 11520]
LEADS = [(0, 60), (60, 240), (240, 720), (720, 1440)] if k == 5 else [(0, 10), (10, 30), (30, 60)]
res = {}
for a in assets:
    df = load_all(a, start='2025-12-01' if a in ('BTC','ETH','SOL','XRP','HYPE') else None)
    lp = np.log(df['close'].to_numpy())
    ny = df.index.tz_convert('America/New_York')
    mow = (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()
    r = np.full(len(lp), np.nan); r[:-k] = lp[k:] - lp[:-k]
    z = r ** 2 / (k * np.maximum(prof[f'{a}_{kind}'][mow], 0.03))
    zs = pd.Series(z)
    med = zs.rolling(1440 * 7, min_periods=1440).median()
    zc = np.minimum(z, 50 * med.to_numpy())   # tame outliers
    feats = {h: pd.Series(zc).ewm(halflife=h, ignore_na=True, min_periods=60).mean().to_numpy() for h in HLS}
    # origin t uses returns ending before t: r at index m covers [m, m+k), known at m+k
    origins = np.arange(30 * 1440, len(lp) - 1440 - k, 60)
    zf = pd.Series(z).fillna(np.nan)
    cs = np.concatenate([[0], np.nancumsum(z)]); cn = np.concatenate([[0], np.cumsum(np.isfinite(z))])
    out = {}
    for (l0, l1) in LEADS:
        tgt = (cs[origins + l1] - cs[origins + l0]) / np.maximum(cn[origins + l1] - cn[origins + l0], 1)
        X = {h: feats[h][origins - k] for h in HLS}
        ok = np.isfinite(tgt) & (tgt > 0) & np.all([np.isfinite(X[h]) & (X[h] > 0) for h in HLS], axis=0)
        best = None
        for h1, h2 in itertools.combinations_with_replacement(HLS, 2):
            for w in np.linspace(0, 1, 11):
                f = w * X[h1][ok] + (1 - w) * X[h2][ok]
                for sc in (0.8, 0.9, 1.0, 1.1):
                    q = np.mean(tgt[ok] / (sc * f) - np.log(tgt[ok] / (sc * f)) - 1)
                    if best is None or q < best[0]:
                        best = (q, h1, h2, w, sc)
        single = min((np.mean(tgt[ok] / (s * X[h][ok]) - np.log(tgt[ok] / (s * X[h][ok])) - 1), h, s) for h in HLS for s in (0.8, 0.9, 1.0, 1.1))
        out[f'{l0}-{l1}'] = f"best {best[1]}/{best[2]} w={best[3]:.1f} sc={best[4]} q={best[0]:.4f} | single {single[1]} sc={single[2]} q={single[0]:.4f}"
    res[a] = out
    print(a, out, flush=True)
