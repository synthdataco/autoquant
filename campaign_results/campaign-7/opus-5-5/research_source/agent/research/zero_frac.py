import sys, numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS
for a in ALL_ASSETS:
    df = load_all(a, start='2026-08-01')
    c = df['close'].to_numpy()
    d1 = np.diff(c); d5 = c[5:] - c[:-5]
    ok1 = np.isfinite(d1); ok5 = np.isfinite(d5)
    nz = np.abs(d1[ok1 & (d1 != 0)])
    tick = np.min(nz) if nz.size else np.nan
    print(f"{a:7s} zero1={np.mean(d1[ok1]==0):.3f} zero5={np.mean(d5[ok5]==0):.3f} nan={np.mean(~np.isfinite(c)):.4f} "
          f"tc0={np.mean(df['trade_count'].to_numpy()==0):.3f} last={c[np.isfinite(c)][-1]:.4f} mintick={tick:.6g} "
          f"tick_bps={1e4*tick/c[np.isfinite(c)][-1]:.3f}")
