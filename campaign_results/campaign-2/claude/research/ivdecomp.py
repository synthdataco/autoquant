"""Which SCORING INTERVALS actually carry each competition's CRPS -- and are we losing it evenly?

I have been treating total CRPS as the thing to minimize and diagnosing it with a PIT built on
non-overlapping interval changes. Reading the validator's crps_calculation source closely shows that
is only part of what it scores, and for crypto-1h it is the minority part:

  crypto-1h scores 18 intervals. Six are ordinary ones (1,2,5,15,30 min plus 60min_abs). The other
  TWELVE are `0_Xmin_gaps` for X = 5,10,...,60. A `_gaps` interval is NOT a series of increments --
  calculate_price_changes_over_intervals truncates to interval_prices[:, :2], so each one scores a
  SINGLE ANCHORED change from the path's start to X minutes in. They are then summed into one "Gaps"
  total. crypto-24h and com-equ-24h have no gaps intervals at all: just 5min, 30min, 3hour, 24hour_abs.

So crypto-1h -- the largest of my three score terms (0.1732) -- is scored substantially on the
anchored cumulative return at twelve horizons, a construction my PIT never looked at. Those twelve
are also strongly mutually correlated (they share one anchor and nest inside each other), so a
calibration error in the anchored distribution is not diversified away: it is charged twelve times.
That is a plausible place for a systematic error to hide, and I have never measured it.

This script decomposes total CRPS by interval name using the validator's own
calculate_crps_for_miner detail output -- no reimplemented scoring anywhere. It answers:

  1. What FRACTION of each competition's CRPS comes from each interval, and specifically from Gaps?
     That is the weight map telling me where a given percentage improvement is worth the most.
  2. Is our CRPS concentrated differently than the geometry alone implies?

An interval carrying a large share is where model effort pays; a large share I have never diagnosed
is where an undetected defect would be most costly. Reported per asset, since the competitions are
scored per asset and then aggregated.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling
from synth.validator.crps_calculation import calculate_crps_for_miner

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='')
ap.add_argument('--start', default='2026-07-16')
ap.add_argument('--end', default='2026-07-26')
ap.add_argument('--every-min', type=int, default=60)
ap.add_argument('--nsim', type=int, default=400)
a = ap.parse_args()

comp = harness.COMPS[a.slug]
assets = (a.assets or ','.join(comp.asset_list)).split(',')
TI, TL = comp.time_increment, comp.time_length
starts = pd.date_range(a.start, a.end, freq=f'{a.every_min}min', tz='UTC')

print(f"CRPS decomposition by scoring interval -- {a.slug}, {a.start}->{a.end}, nsim={a.nsim}")
print("(validator's own calculate_crps_for_miner detail output; 'Gaps' = the 12 anchored intervals)\n")

allrows = []
for asset in assets:
    tot = {}
    n = 0
    for t in starts:
        try:
            ctx = harness.context(asset, t)
            if ctx is None or len(ctx) < 2000:
                continue
            res = modeling.simulate(asset=asset, start_time=t, time_increment=TI,
                                    time_length=TL, num_simulations=a.nsim, context_prices=ctx)
            paths = np.asarray(res[2:], dtype=float)
            real = harness.realized(asset, t, TL, TI)
            if real is None or len(real) != paths.shape[1] or not np.isfinite(real).all():
                continue
            _, detail = calculate_crps_for_miner(paths, np.asarray(real), TI, comp.scoring_intervals)
        except Exception:
            continue
        for d in detail:
            if 'Interval' not in d:
                continue
            tot[d['Interval']] = tot.get(d['Interval'], 0.0) + float(d['CRPS'])
        n += 1
    if not n or 'Overall' not in tot:
        print(f"{asset:8s} no results"); continue
    overall = tot.pop('Overall')
    print(f"{asset:8s} n={n} prompts, mean total CRPS = {overall / n:.2f}")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14s}{v / n:10.2f}{v / overall * 100:8.1f}% of total")
        allrows.append(dict(asset=asset, iv=k, share=v / overall * 100))
    print()

if allrows:
    d = pd.DataFrame(allrows)
    print("=" * 60)
    print("mean share of total CRPS by interval, across assets:")
    print(d.groupby('iv').share.mean().sort_values(ascending=False).round(1).to_string())
    print("\nThe largest share is where a 1% CRPS improvement buys the most. If 'Gaps' dominates,")
    print("the anchored-return distribution -- not the increment distribution -- is what to calibrate.")
