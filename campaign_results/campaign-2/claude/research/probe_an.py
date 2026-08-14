"""Analyse the per-prompt probe frame: is the deficit uniform-multiplicative or state-dependent?

THE TEST. Regress per-prompt excess (ours - leadN) on the leader's own CRPS:

    excess_i = a + b * leadN_i + e_i

A purely UNIFORM multiplicative defect (ours = c*leader everywhere) implies a = 0 and b = c-1, and
the fit absorbs essentially all the structure. A STATE-DEPENDENT defect leaves the residual e_i
correlated with some variable known at prompt time -- which is then a named, actionable channel.

So after the regression we check what the residual still correlates with, separating variables
known at prompt time (candidate model inputs) from realized outcomes (diagnosis only).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--path", default="/workspace/agent/artifacts/probe_1h.parquet")
ap.add_argument("--assets", default="")
a = ap.parse_args()

df = pd.read_parquet(a.path)
if a.assets:
    df = df[df.asset.isin(a.assets.split(","))]

STATE = ["sigma_min", "rv_1h", "rv_1d", "rv_7d", "vr_1h_1d", "vr_1d_7d"]   # known at prompt time
OUTCOME = ["z_move", "vol_surprise", "max_step_z"]                          # diagnosis only

for asset, d in df.groupby("asset"):
    d = d.dropna(subset=["ours", "leadN"]).copy()
    if len(d) < 50:
        continue
    d["excess"] = d.ours - d.leadN
    gap = d.ours.mean() / d.leadN.mean() - 1

    # --- the uniform-multiplicative test ---
    x = d.leadN.to_numpy()
    y = d.excess.to_numpy()
    b, a0 = np.polyfit(x, y, 1)
    pred = a0 + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Under a uniform defect the slope should equal the overall gap and the intercept should be 0
    # relative to the CRPS scale; report the intercept as a fraction of mean leader CRPS.
    print(f"=== {asset}  n={len(d)}  gap {gap*100:+.3f}%  rank {int(d.rank_all.iloc[0])}/{int(d.n_panel.iloc[0])} ===")
    print(f"  excess ~ a + b*leader :  b = {b:+.5f} (uniform predicts {gap:+.5f})"
          f"   a = {a0:+.4f} ({a0/d.leadN.mean()*100:+.3f}% of mean leader CRPS)   R2 = {r2:.4f}")

    d["resid"] = y - pred
    # z-scored residual correlations, so magnitudes are comparable across variables
    cors = []
    for c in STATE + OUTCOME:
        if c not in d.columns:
            continue
        v = d[c].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(d.resid.to_numpy())
        if m.sum() < 50 or np.std(v[m]) == 0:
            continue
        r = float(np.corrcoef(v[m], d.resid.to_numpy()[m])[0, 1])
        cors.append((c, r, "state" if c in STATE else "outcome"))
    cors.sort(key=lambda z: -abs(z[1]))
    print("  residual correlations: " + "  ".join(f"{c}={r:+.3f}[{k[0]}]" for c, r, k in cors))

    # quintile view of the RATIO on the strongest state variable and on the outcomes
    for c in ["vol_surprise", "max_step_z", "vr_1h_1d"]:
        if c not in d.columns:
            continue
        v = d[c]
        if v.notna().sum() < 100:
            continue
        q = pd.qcut(v, 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "ratio": d.ours / d.leadN, "exc": d.excess}).groupby("q")
        rr = g["ratio"].mean()
        sh = g["exc"].sum() / d.excess.sum()
        print(f"    {c:>13s}  ratio: " + " ".join(f"{x:.4f}" for x in rr)
              + "   excess share: " + " ".join(f"{x*100:5.1f}%" for x in sh))
    print()

# pooled view: how much of the TOTAL excess (coefficient-weighted, i.e. what the validator pays on)
# comes from each asset
try:
    from synth.validator.moving_average import ASSET_COEFFICIENTS as AC
    g = df.groupby("asset").agg(ours=("ours", "mean"), leadN=("leadN", "mean"))
    g["gap_pct"] = (g.ours / g.leadN - 1) * 100
    g["coef"] = [AC.get(x, 1.0) for x in g.index]
    # the smoothed score is a coefficient-weighted MEAN of prompt_score = crps - field_min, so an
    # asset's contribution to the score gap scales with coef * its own CRPS excess
    g["excess"] = g.ours - g.leadN
    g["contrib"] = g.coef * g.excess
    g["contrib_share"] = g.contrib / g.contrib.sum() * 100
    print("=== coefficient-weighted contribution to the smoothed-score gap ===")
    print(g.sort_values("contrib", ascending=False).round(4).to_string())
except Exception as e:  # pragma: no cover
    print("coef view unavailable:", e)
