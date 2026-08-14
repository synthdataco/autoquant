"""NVDAX/GOOGLX sit at +11% while TSLAX (same class, same code, same window) sits at +0.6%. Why?

Everything global has now been ruled out for the com-equ deficit:
  * dispersion LEVEL -- a scale sweep over 0.85..1.08 moves the mean 0.8% against an 11.9% gap.
  * seasonal amplitude -- season_pow sweep puts the optimum exactly at the current 1.0.
  * innovation SHAPE by session -- fixed, worth 0.33-0.47%, but NVDAX stays at +11.3%.
  * a few bad prompts -- equwhen.py: median prompt ratio ~= mean, worst 10% carry only 22-24%.
  * field sharpness -- equsplit.py: corr(field spread, our gap) = -0.07.

A UNIFORM 11% CRPS excess with a correctly-shaped, correctly-scaled ensemble is a strong signal,
because for a roughly-Gaussian forecast CRPS is near-quadratic around the optimum: being 11% worse
on EVERY prompt while the dispersion level is provably near-optimal is very hard to achieve by
mis-sizing the fan. The remaining ways to be uniformly wrong are about the CENTRE, not the width:

  (H1) DRIFT / anchor error. We start every path at the last context price and add zero-mean
       increments. If this asset's feed has a persistent basis (a tokenized equity trading at a
       premium that decays, or a stale last print that systematically reverts toward a reference),
       then the true conditional mean is NOT the last price, and every prompt pays for it. This is
       the classic way to be uniformly bad while perfectly calibrated in width.

  (H2) The last context price is a BAD ANCHOR specifically because it is stale. If the feed's final
       print is minutes old and the true price has moved on, we are anchored to a lagged value.
       This would hurt most where the feed is stale in a directional way.

Both are testable with no modelling: measure the realized log change from the anchor over the scored
horizons and ask whether its mean is reliably non-zero, and whether the ensemble's median sits where
the realizations centre. The script reports, per asset, over prompt start times in the window:

  * mean realized drift at 5min / 1h / 24h, in bp, with a t-statistic. A |t| > 3 on 100+ prompts is
    a real conditional mean we are ignoring.
  * the same for TSLAX and BTC as controls -- if TSLAX (which we score well on) has no drift and
    NVDAX does, that is the whole explanation.
  * the CRPS cost implied by that drift: for a Gaussian ensemble, shifting the centre by d when the
    sd is s inflates CRPS by roughly 0.23*(d/s)^2*s for small d/s -- reported as a % of our current
    CRPS so it can be compared directly against the 11% gap.

Uses ONLY snapshot prices at times <= the prompt start for the anchor, and the realized path after
it purely as the scoring target, exactly as the validator does. No lookahead beyond 2026-07-27.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='NVDAX,GOOGLX,XAU,TSLAX,AAPLX,SPCX,WTIOIL,BTC')
ap.add_argument('--start', default='2026-07-16')
ap.add_argument('--end', default='2026-07-26')
a = ap.parse_args()

print("realized drift from the anchor (last context price), by horizon")
print("a large |t| means the true conditional mean is not the last price -- a centre error,")
print("which is the only cheap way left to be uniformly ~11% worse at a near-optimal width\n")
print(f"{'asset':8s}{'n':>5s}{'5min bp':>10s}{'t':>7s}{'1h bp':>10s}{'t':>7s}"
      f"{'24h bp':>10s}{'t':>7s}{'sd24h bp':>10s}{'implied':>9s}")

for asset in a.assets.split(','):
    try:
        s = harness.load_prices(asset)
    except Exception as e:
        print(f"{asset:8s} {e}"); continue
    s = s[np.isfinite(s.values)]
    starts = pd.date_range(a.start, a.end, freq='h', tz='UTC')
    rows = []
    for t in starts:
        # anchor: last observed price at or before t (this is what simulate() uses)
        try:
            anchor_idx = s.index.searchsorted(t, side='right') - 1
            if anchor_idx < 0:
                continue
            p0 = float(s.iloc[anchor_idx])
            if not np.isfinite(p0) or p0 <= 0:
                continue
            d = {}
            for name, mins in [('m5', 5), ('h1', 60), ('h24', 1440)]:
                j = s.index.searchsorted(t + pd.Timedelta(minutes=mins), side='right') - 1
                if j <= anchor_idx or j >= len(s):
                    d = None; break
                p1 = float(s.iloc[j])
                if not np.isfinite(p1) or p1 <= 0:
                    d = None; break
                d[name] = np.log(p1 / p0) * 1e4
            if d:
                rows.append(d)
        except Exception:
            continue
    if len(rows) < 30:
        print(f"{asset:8s} only {len(rows)} usable starts"); continue
    df = pd.DataFrame(rows)
    out = f"{asset:8s}{len(df):5d}"
    for name in ['m5', 'h1', 'h24']:
        v = df[name].to_numpy()
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        t_ = mu / (sd / np.sqrt(len(v))) if sd > 0 else np.nan
        out += f"{mu:10.2f}{t_:7.2f}"
    sd24 = float(np.std(df['h24'], ddof=1))
    mu24 = float(np.mean(df['h24']))
    # CRPS inflation from an off-centre ensemble, Gaussian approx, as a fraction of CRPS itself
    ratio = abs(mu24) / sd24 if sd24 > 0 else np.nan
    implied = 0.23 * ratio ** 2 / 0.234 * 100 if np.isfinite(ratio) else np.nan
    out += f"{sd24:10.1f}{implied:8.2f}%"
    print(out)

print("\n'implied' = the CRPS excess a Gaussian ensemble would suffer at 24h purely from ignoring")
print("that drift. Compare against the measured gap (NVDAX +11.3%, GOOGLX +11.0%, TSLAX +0.6%).")
print("If implied is ~0 everywhere, the centre is fine and the deficit is NOT a drift problem --")
print("which would leave the field simply being better on these assets by a route we have not found.")
