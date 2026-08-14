"""HYPE's gap slides monotonically from +4.1% to -1.3% across six days. What is actually moving?

hype.py killed the rv_7d story (lag-1 autocorr 0.9996, ~0.86 independent observations, correlation
collapses +0.304 -> +0.037 detrended). But the per-day gap is not noise-shaped: it runs
+3.20, +4.07, +3.26, +1.90, -0.26, -1.33 -- monotone after day 2 and crossing zero. Something is
drifting, and there are two very different candidates:

  (A) OUR dispersion is mis-levelled and the market's vol is trending, so a fixed kcal is too wide
      early and about right late (or vice versa). This is a LEVEL error: fixable, and the fix is a
      faster-adapting sigma rather than a different constant.
  (B) The leaders adapted to something we did not (a HYPE-specific event, a listing change, a
      regime), in which case the early days are not informative about the evaluation window.

The discriminator is our own calibration measured against the REALIZED outcome per day, which is
independent of what the leaders did:
   vol_surprise = realized step RV / our forecast step sigma.
If (A), vol_surprise should track the gap across days -- we lose on days we mis-size the fan.
If (B), our calibration can be flat while the gap moves, i.e. the leaders simply got better.

This matters for the champion because the evaluation window is the 10 days AFTER the snapshot: if
(A), the right response is a model change; if (B), the last two days are the best estimate of our
standing and the early days are stale.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="/workspace/agent/artifacts/probe_1h.parquet")
ap.add_argument("--assets", default="HYPE,BTC,ETH,SOL,XRP")
a = ap.parse_args()

df = pd.read_parquet(a.path)
df["start_time"] = pd.to_datetime(df.start_time, utc=True)
df["day"] = df.start_time.dt.floor("D")

for asset in a.assets.split(","):
    d = df[df.asset == asset].dropna(subset=["ours", "leadN"]).copy()
    if len(d) < 100:
        continue
    g = d.groupby("day").agg(
        n=("ours", "size"), ours=("ours", "mean"), lead=("leadN", "mean"),
        # our own calibration vs the realized outcome -- independent of the field
        vsurp=("vol_surprise", "median"), zsd=("z_move", "std"),
        maxz=("max_step_z", "median"), sig1=("sig_1", "mean"),
    )
    g["gap_pct"] = (g.ours / g.lead - 1) * 100
    g["sig1_bp"] = g.sig1 * 1e4
    print(f"=== {asset} ===")
    print(g[["n", "gap_pct", "vsurp", "zsd", "maxz", "sig1_bp"]].round(4).to_string())
    # correlation across days between our mis-calibration and our gap to the leaders
    if len(g) >= 4:
        for c in ["vsurp", "zsd", "maxz"]:
            v, w = g[c].to_numpy(), g.gap_pct.to_numpy()
            m = np.isfinite(v) & np.isfinite(w)
            if m.sum() >= 4 and np.std(v[m]) > 0:
                print(f"   corr(day gap, {c}) = {np.corrcoef(v[m], w[m])[0,1]:+.3f}  (n_days={m.sum()})")
    print()
