"""Variance ratios at the 24h prompt's scales, robust (|r| based): E|r_k| / (sqrt(k/5) E|r_5|) on
non-overlapping k-min returns, k = 30, 180, 1440, last 180 days, deseasonalised by the a5 profile
only through the ratio of sums (approximate)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS
for a in ALL_ASSETS:
    df = load_all(a, start='2026-05-01')
    lp = np.log(df['close']).to_numpy()
    out = []
    for k in (30, 180, 1440):
        vals = []
        for off in range(0, k, max(k // 6, 5)):
            idx = np.arange(off, len(lp) - k, k)
            rk = lp[idx + k] - lp[idx]
            # the 5-min returns that tile the same k-min windows
            r5 = np.concatenate([lp[idx + j + 5] - lp[idx + j] for j in range(0, k, 5)])
            ok = np.isfinite(rk); ok5 = np.isfinite(r5)
            vals.append(np.mean(rk[ok] ** 2) / (k / 5 * np.mean(r5[ok5] ** 2)))
        out.append(np.median(vals))
    print(f"{a:7s} VR(5->30) {out[0]:.3f}  VR(5->180) {out[1]:.3f}  VR(5->1440) {out[2]:.3f}")
