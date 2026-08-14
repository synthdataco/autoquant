"""HYPE carries 51% of the crypto-1h score gap. Is its state-dependence real, or is it TIME?

probe_an.py found HYPE alone has a genuinely structured deficit: excess ~ leader CRPS gives
R^2 = 0.164 (every other asset 0.005-0.046) and the residual correlates +0.304 with trailing 7-day
realized vol -- by far the strongest state correlation anywhere in the competition.

THE CONFOUND, and why it must be checked first. rv_7d is a 7-day trailing average sampled over a
6-day window, so consecutive prompts share ~99% of their averaging window. Its quintiles are
therefore close to a partition of TIME, and "the residual correlates with rv_7d" is nearly
indistinguishable from "we drifted relative to the leaders during the window" -- which would be a
regime story with an effective sample size of about 1, not an actionable conditional model.

Three checks, in increasing strength:
  1. correlation of the residual with time itself, and with rv_7d after removing a linear time trend;
  2. effective sample size: how many INDEPENDENT rv_7d observations does the window contain?
  3. does the relationship hold WITHIN days (rv_7d demeaned per day), where time is controlled?

Only if the effect survives (3) is there a conditional channel worth modelling.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="/workspace/agent/artifacts/probe_1h.parquet")
ap.add_argument("--asset", default="HYPE")
a = ap.parse_args()

d = pd.read_parquet(a.path)
d = d[d.asset == a.asset].dropna(subset=["ours", "leadN"]).copy()
d["start_time"] = pd.to_datetime(d.start_time, utc=True)
d = d.sort_values("start_time").reset_index(drop=True)
d["excess"] = d.ours - d.leadN
d["ratio"] = d.ours / d.leadN

x = d.leadN.to_numpy()
b, a0 = np.polyfit(x, d.excess.to_numpy(), 1)
d["resid"] = d.excess.to_numpy() - (a0 + b * x)

t = (d.start_time - d.start_time.min()).dt.total_seconds().to_numpy() / 86400.0
d["tday"] = t


def cor(u, v) -> float:
    m = np.isfinite(u) & np.isfinite(v)
    if m.sum() < 30 or np.std(u[m]) == 0 or np.std(v[m]) == 0:
        return np.nan
    return float(np.corrcoef(u[m], v[m])[0, 1])


print(f"=== {a.asset}  n={len(d)}  gap {(d.ours.mean()/d.leadN.mean()-1)*100:+.3f}% ===\n")

# ---- 1. is it just time? ----
print("1) TIME CONFOUND")
print(f"   corr(resid, time)           = {cor(d.resid.to_numpy(), t):+.3f}")
print(f"   corr(resid, rv_7d)          = {cor(d.resid.to_numpy(), d.rv_7d.to_numpy()):+.3f}")
print(f"   corr(rv_7d, time)           = {cor(d.rv_7d.to_numpy(), t):+.3f}   <-- if ~|1|, rv_7d IS time")
# residual of rv_7d after removing a linear time trend, vs the CRPS residual
bt_, at_ = np.polyfit(t, d.rv_7d.to_numpy(), 1)
rv_detr = d.rv_7d.to_numpy() - (at_ + bt_ * t)
print(f"   corr(resid, rv_7d | detrended time) = {cor(d.resid.to_numpy(), rv_detr):+.3f}")

# ---- 2. effective sample size ----
print("\n2) EFFECTIVE SAMPLE SIZE")
span_days = t.max() - t.min()
print(f"   window spans {span_days:.2f} days; rv_7d averages over 7 days")
print(f"   => roughly {span_days/7:.2f} independent rv_7d observations")
r1 = cor(d.rv_7d.to_numpy()[1:], d.rv_7d.to_numpy()[:-1])
print(f"   lag-1 autocorrelation of rv_7d across consecutive prompts = {r1:+.4f}")

# ---- 3. within-day test (time controlled) ----
print("\n3) WITHIN-DAY TEST (does it survive controlling for the day?)")
d["day"] = d.start_time.dt.floor("D")
for col in ["rv_7d", "rv_1d", "rv_1h", "vr_1h_1d"]:
    dd = d.dropna(subset=[col]).copy()
    if len(dd) < 100:
        continue
    dd["x_dm"] = dd[col] - dd.groupby("day")[col].transform("mean")
    dd["r_dm"] = dd.resid - dd.groupby("day")["resid"].transform("mean")
    c_raw = cor(dd[col].to_numpy(), dd.resid.to_numpy())
    c_dm = cor(dd.x_dm.to_numpy(), dd.r_dm.to_numpy())
    print(f"   {col:>10s}: raw {c_raw:+.3f}   within-day {c_dm:+.3f}")

# ---- per-day view: is the deficit concentrated in particular days? ----
print("\n4) PER-DAY BREAKDOWN")
g = d.groupby("day").agg(n=("ours", "size"), ours=("ours", "mean"), lead=("leadN", "mean"),
                         rv7=("rv_7d", "mean"), rv1d=("rv_1d", "mean"))
g["gap_pct"] = (g.ours / g.lead - 1) * 100
g["rv7_bp"] = g.rv7 * 1e4
g["rv1d_bp"] = g.rv1d * 1e4
print(g[["n", "ours", "lead", "gap_pct", "rv7_bp", "rv1d_bp"]].round(3).to_string())
