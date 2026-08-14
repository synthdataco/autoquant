"""Per-asset dispersion recalibration for com-equ, on the theory that the assets disagree.

Why a per-asset fit when a global `scale` sweep was flat. pit.py shows the com-equ assets are
mis-dispersed in OPPOSITE DIRECTIONS at the horizons that carry the score:

  NVDAX   30min PIT 12.0/23.7/26.8/22.0/13.4  -> a clean central hump: TOO WIDE, at every
          horizon from 30min out to 24h (180min is 12.0/22.3/37.7/17.6/10.4, badly humped)
  GOOGLX  24h  PIT 37.5/18.8/15.6/14.1/14.1, 12.5% outside support -> TOO NARROW
  TSLAX   24h  PIT 51.5/15.2/ 4.5/13.6/15.2, 15.2% outside support -> TOO NARROW, severely

A single global multiplier cannot fix a set of assets that need opposite corrections; it averages
them and lands near 1.0, which is exactly what the scale sweep reported (0.85->1.08 moved the mean
0.8% and was nearly flat). That flat sweep was not evidence that dispersion is right -- it was two
real errors cancelling in the mean. This is the same lesson as the horizon profile in iteration 14,
one level up: aggregate diagnostics hide sign-flipping structure.

NVDAX being too wide also finally explains its +11.3% gap, the largest and most stubborn in the
competition, in a way that every previous hypothesis failed to: a hump at EVERY horizon is a uniform
per-prompt cost (matching equwhen.py's finding that the median prompt ratio equals the mean), it is
invisible to a drift test (nvdiag.py, correctly refuted), and it is untouchable by better vol
FORECASTING (oracle.py: perfect foresight worth 0.57%) because it is a level error, not a
state-tracking error.

Method. Fit a per-asset multiplier on the forecast sigma by direct CRPS search on a TRAINING window,
then report the held-out window separately. `kcal` already exists for exactly this purpose (per-asset,
per-increment CRPS-optimal dispersion) and was fitted on 07-08..07-20; the hypothesis is that its
equity values are stale or were fitted against the pre-session-pool model. So this refits the same
object with the current model and checks whether the correction REPLICATES out of sample -- the test
that iteration 17's kcal refit failed, when two diagnostics agreed only because they shared a window.

Nothing is written to any artifact here. This only reports, so the decision to adopt is made against
the holdout column, not against the fit.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import modeling
from harness import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='com-equ-24h')
ap.add_argument('--assets', default='NVDAX,GOOGLX,TSLAX,AAPLX,XAU,WTIOIL,SPCX')
ap.add_argument('--fit', default='2026-07-16:2026-07-21')
ap.add_argument('--holdout', default='2026-07-21:2026-07-26')
ap.add_argument('--ks', default='0.88,0.94,1.0,1.06,1.12')
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--stride', type=int, default=2)
a = ap.parse_args()

ks = [float(x) for x in a.ks.split(',')]
fs, fe = a.fit.split(':')
hs, he = a.holdout.split(':')

print(f"per-asset dispersion multiplier, fit {a.fit} / holdout {a.holdout}, nsim={a.nsim}")
print("adopt ONLY where the fitted k reduces CRPS on the HOLDOUT too (iteration 17's lesson)\n")
print(f"{'asset':8s}{'k*(fit)':>9s}{'fit@1.0':>9s}{'fit@k*':>9s}{'gain':>8s}"
      f"{'hold@1.0':>10s}{'hold@k*':>9s}{'gain':>8s}   verdict")

for asset in a.assets.split(','):
    fit_res, hold_res = {}, {}
    for k in ks:
        P = {"scale": k}
        for tag, (s, e), store in (("f", (fs, fe), fit_res), ("h", (hs, he), hold_res)):
            try:
                r = evaluate(modeling.simulate, a.slug, asset, s, e,
                             nsim=a.nsim, workers=13, stride=a.stride, params=P)
                if len(r):
                    store[k] = float((r.crps.mean() / r.field_min.mean() - 1) * 100)
            except Exception as ex:
                print(f"  {asset} k={k} {tag} failed: {type(ex).__name__} {ex}")
    if 1.0 not in fit_res or not fit_res or not hold_res:
        print(f"{asset:8s} insufficient results"); continue
    kstar = min(fit_res, key=fit_res.get)
    fg = fit_res[1.0] - fit_res[kstar]
    hg = hold_res.get(1.0, np.nan) - hold_res.get(kstar, np.nan)
    if kstar == 1.0:
        v = "no change"
    elif np.isfinite(hg) and hg > 0.05:
        v = "ADOPT (replicates)"
    elif np.isfinite(hg) and hg > -0.05:
        v = "neutral out of sample"
    else:
        v = "REJECT (does not replicate)"
    print(f"{asset:8s}{kstar:9.2f}{fit_res[1.0]:9.2f}{fit_res[kstar]:9.2f}{fg:+8.2f}"
          f"{hold_res.get(1.0, np.nan):10.2f}{hold_res.get(kstar, np.nan):9.2f}{hg:+8.2f}   {v}")

print("\nA k* far from 1.0 that gains on fit but not holdout is overfitting to one window's realized")
print("vol -- the exact failure mode of iteration 17. Only the replicating ones are real.")
