"""Exploration: does recent trade activity (trade_count) predict the next hour's deseasonalised
|r| beyond the HAR forecast? OLS of log(next-60min mean |r|/sqrt(a1)) on log(HAR forecast) and
log(trade-count fast/slow ratio). Data 2026-06-01..2026-10-01, origins every 20 min."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
prof = np.load('agent/research/profilesxm_2026-10-01.npz')
for a in ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE']:
    df = load_all(a, start='2026-05-01', end='2026-10-01 00:00')
    lp = np.log(df['close'].to_numpy())
    r = np.full(len(lp), np.nan); r[:-1] = lp[1:] - lp[:-1]
    ny = df.index.tz_convert('America/New_York')
    mow = (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()
    pa = np.maximum(prof[f'{a}_a1'][mow], 0.03)
    z = np.abs(r) / np.sqrt(pa)
    tc = df['trade_count'].to_numpy(dtype=float)
    # trade-count seasonal profile (mean per minute of week over the sample, normalised)
    tprof = pd.Series(tc).groupby(mow).mean().reindex(range(10080)).to_numpy()
    tprof = tprof / np.nanmean(tprof)
    tcd = tc / np.maximum(tprof[mow], 1e-6)
    E = lambda x, h: pd.Series(x).ewm(halflife=h, ignore_na=True, min_periods=30).mean().to_numpy()
    e10, e60, e480 = E(z, 10), E(z, 60), E(z, 480)
    t10, t480 = E(tcd, 10), E(tcd, 480)
    o = np.arange(31 * 1440, len(lp) - 62, 20)
    f = 0.3 * e10[o - 2] + 0.25 * e60[o - 2] + 0.45 * e480[o - 2]
    cs = np.concatenate([[0], np.nancumsum(z)]); cn = np.concatenate([[0], np.cumsum(np.isfinite(z))])
    y = (cs[o + 61] - cs[o + 1]) / np.maximum(cn[o + 61] - cn[o + 1], 1)
    x2 = np.log(t10[o - 2] / t480[o - 2])
    x3 = np.log(e10[o - 2] / e480[o - 2])
    ok = np.isfinite(y) & (y > 0) & np.isfinite(f) & (f > 0) & np.isfinite(x2) & np.isfinite(x3)
    Y = np.log(y[ok]); X1 = np.log(f[ok])
    A1 = np.column_stack([np.ones(ok.sum()), X1])
    A2 = np.column_stack([np.ones(ok.sum()), X1, x2[ok]])
    b1, res1 = np.linalg.lstsq(A1, Y, rcond=None)[:2]
    b2, res2 = np.linalg.lstsq(A2, Y, rcond=None)[:2]
    r2 = lambda res: 1 - res[0] / np.sum((Y - Y.mean()) ** 2)
    print(f"{a}: n={ok.sum()} HAR-only slope {b1[1]:.3f} R2 {r2(res1):.4f} | +tc: slope {b2[1]:.3f} tc-coef {b2[2]:.3f} R2 {r2(res2):.4f} | corr(tc ratio, vol ratio) {np.corrcoef(x2[ok], x3[ok])[0,1]:.2f}")
