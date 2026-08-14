"""Fit the multiplicative seasonality + build the standardized-residual pools.

Writes agent/artifacts/calib.npz — used by modeling.py at serve time (fully offline).
Estimation uses ONLY the snapshot (<= 2026-07-27), which the constitution allows for fitting.
"""
import numpy as np, pandas as pd, sys, json
from pathlib import Path
sys.path.insert(0,'/workspace/agent/research')
from harness import load_prices

ASSETS = ["BTC","ETH","SOL","XRP","HYPE","XAU","NVDAX","GOOGLX","TSLAX","AAPLX","WTIOIL","SPCX","SP500"]
OUT = Path('/workspace/agent/artifacts'); OUT.mkdir(exist_ok=True)

def minute_returns(asset):
    s = load_prices(asset)
    r = pd.Series(np.diff(np.log(s.values)), index=s.index[1:])
    return s, r[np.isfinite(r)]

cal = {}
for a in ASSETS:
    s, r = minute_returns(a)
    ab = r.abs()
    # ---- seasonality on the MINUTE grid: 168 hour-of-week buckets, smoothed ----
    how = r.index.dayofweek*24 + r.index.hour
    # robust scale per bucket: use mean|r| (scaled) rather than std -> less tail-driven
    g = ab.groupby(how).mean()
    g = g.reindex(range(168)).interpolate().bfill().ffill()
    # circular smoothing over +-1 hour
    v = g.values
    sm = (np.roll(v,1)+2*v+np.roll(v,-1))/4.0
    season = sm/sm.mean()
    season = np.clip(season, 0.25, 3.0)
    # ---- deseasonalized standardized residual pool ----
    fac = pd.Series(season[how], index=r.index)
    rd = r/fac
    cal[a] = dict(season=season,
                  sd_minute=float(rd.std()),
                  n=len(rd))
    print(f"{a:7s} season min={season.min():.2f} max={season.max():.2f} "
          f"(weekend mean={season[120:168].mean():.2f}, weekday={season[0:120].mean():.2f})  "
          f"deseas sd={rd.std()*1e4:.2f}bp kurt={rd.kurtosis():.0f}")

np.savez_compressed(OUT/'season.npz', **{f"{a}_season": cal[a]['season'] for a in cal},
                    **{f"{a}_sd": np.array([cal[a]['sd_minute']]) for a in cal})
print("\nwrote", OUT/'season.npz')
