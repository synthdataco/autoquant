"""WHAT is wrong with HYPE? Per-interval CRPS decomposition + PIT, against the field.

HYPE is 57% of the crypto-1h coefficient-weighted score gap (+3.10% over 30 days, against 0.45-0.97%
for the other four assets) and loses on 17 of 19 days. That is a structural defect, not a window.

This script asks WHERE in the score it lives, using the validator's own per-interval detail output:

  * per-interval CRPS for HYPE vs the same decomposition for BTC (the control). If HYPE's excess is
    concentrated in the 1min marginal it is a per-step dispersion/shape problem; if it is in the
    Gaps/cumulative intervals it is an autocovariance problem; if it is spread proportionally it is
    a pure level error.
  * PIT (rank of the realized value inside the ensemble) at each horizon. Uniform = calibrated;
    a central hump = too wide; U-shaped/mass at the edges = too narrow.

Both are computed on the field's own prompt times so the CRPS numbers are directly comparable to the
cached field scores.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/agent/research")
sys.path.insert(0, "/workspace/agent")
import harness  # noqa: E402
from synth.validator.crps_calculation import calculate_crps_for_miner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="crypto-1h")
ap.add_argument("--assets", default="HYPE,BTC")
ap.add_argument("--start", default="2026-07-14")
ap.add_argument("--end", default="2026-07-26")
ap.add_argument("--nsim", type=int, default=1000)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--mod", default="modeling")
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]
TI, TL = comp.time_increment, comp.time_length
STEP_MIN = TI / 60.0

# horizons (in steps) at which to take the PIT, plus the anchored (from start) view
PIT_STEPS = [1, 2, 5, 15, 30, 60]


def one(args):
    asset, t, seed = args
    np.random.seed(seed)
    ctx = harness.context(asset, t)
    if len(ctx) < 2000:
        return None
    real = harness.realized(asset, t, TL, TI)
    if not np.isfinite(real[0]):
        return None
    out = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=TI,
                       time_length=TL, num_simulations=a.nsim, context_prices=ctx)
    paths = np.asarray(out[2:], dtype=float)
    _, detail = calculate_crps_for_miner(paths, real, TI, comp.scoring_intervals)
    d = {r["Interval"]: float(r["CRPS"]) for r in detail if "Interval" in r}

    # --- PIT: anchored change from path start to h steps in (this is what *_gaps scores) ---
    pit = {}
    for h in PIT_STEPS:
        if h >= real.size or not np.isfinite(real[h]):
            continue
        sim_ch = (paths[:, h] - paths[:, 0]) / paths[:, 0]
        real_ch = (real[h] - real[0]) / real[0]
        pit[f"anch{h}"] = float(np.mean(sim_ch < real_ch))
    # --- PIT on the ONE-STEP marginal, pooled over all steps (the 1min interval) ---
    step_pits = []
    for h in range(1, min(real.size, paths.shape[1])):
        if not (np.isfinite(real[h]) and np.isfinite(real[h - 1])):
            continue
        sim_ch = (paths[:, h] - paths[:, h - 1]) / paths[:, h - 1]
        real_ch = (real[h] - real[h - 1]) / real[h - 1]
        step_pits.append(float(np.mean(sim_ch < real_ch)))
    return {"asset": asset, "start_time": t, **{f"crps_{k}": v for k, v in d.items()},
            **pit, "step_pits": step_pits}


from concurrent.futures import ProcessPoolExecutor  # noqa: E402

for asset in a.assets.split(","):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(a.start, tz="UTC"))
            & (fp.start_time < pd.Timestamp(a.end, tz="UTC"))].iloc[::a.stride]
    jobs = [(asset, t, 20_000 + i) for i, t in enumerate(fp.start_time)]
    with ProcessPoolExecutor(max_workers=13) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=1) if r is not None]
    if not rows:
        print(f"{asset}: nothing")
        continue
    df = pd.DataFrame(rows)

    print(f"=== {asset}   n={len(df)} prompts ===")
    ivcols = [c for c in df.columns if c.startswith("crps_")]
    tot = df["crps_Overall"].mean()
    print("  per-interval CRPS (share of Overall):")
    for c in sorted(ivcols, key=lambda z: -df[z].mean()):
        name = c[5:]
        if name == "Overall":
            continue
        print(f"    {name:>12s}  {df[c].mean():9.4f}   {df[c].mean()/tot*100:5.2f}%")
    print(f"    {'Overall':>12s}  {tot:9.4f}")

    print("  anchored PIT (quintile shares; uniform = 20/20/20/20/20, "
          "central hump = too WIDE, edges heavy = too NARROW):")
    for h in PIT_STEPS:
        c = f"anch{h}"
        if c not in df.columns or df[c].notna().sum() < 30:
            continue
        v = df[c].dropna().to_numpy()
        q = np.histogram(v, bins=[0, .2, .4, .6, .8, 1.0001])[0] / len(v) * 100
        outside = float(np.mean((v <= 1e-9) | (v >= 1 - 1e-9))) * 100
        print(f"    {h:>3d} steps  " + " ".join(f"{x:5.1f}" for x in q)
              + f"   outside {outside:4.1f}%   mean {v.mean():.3f}")

    allp = np.concatenate([np.asarray(x) for x in df.step_pits if len(x)])
    q = np.histogram(allp, bins=[0, .2, .4, .6, .8, 1.0001])[0] / len(allp) * 100
    print(f"  ONE-STEP PIT pooled (n={len(allp)}): " + " ".join(f"{x:5.1f}" for x in q)
          + f"   outside {float(np.mean((allp<=1e-9)|(allp>=1-1e-9)))*100:4.1f}%   mean {allp.mean():.3f}")
    print()
