"""PIT (rank histogram) of our ensemble vs the realized path -- where is the marginal actually wrong?

NVDAX sits at +11.3% vs the field best on com-equ-24h with, at this point, everything global ruled
out: dispersion level (flat sweep), seasonal scale (optimum at current), innovation shape by session
(fixed, -0.47%), drift (refuted, wrong sign), a few bad prompts (refuted, uniform shift), field
sharpness (refuted), and now the whole vol channel (oracle bound: perfect foresight worth 0.57%
against an 11.3% gap).

When every parameter is near-optimal and the model is still uniformly beaten, the thing to do is stop
guessing at parameters and ask the ensemble directly where its probability is misplaced. The
probability integral transform does that: for each scored point, compute the rank of the realized
value within the ensemble, i.e. u = P(ensemble <= realized). If the ensemble were perfectly
calibrated, u would be uniform on [0,1] across prompts. Every characteristic deviation names a
specific defect:

  U-shaped  (mass at 0 and 1)  -> ensemble too NARROW, realizations land outside it too often
  hump      (mass at 0.5)      -> ensemble too WIDE
  monotone tilt                -> biased centre (drift), already refuted but worth re-confirming here
  spike at exactly 0.5         -> too much mass at zero (over-applied zero atom)
  asymmetry (skewed histogram) -> the ensemble is symmetric and the truth is not

Crucially this is computed PER HORIZON, because a defect that is invisible in the aggregate can be
severe at the horizons that carry the score weight. And it is computed on the same non-overlapping
interval changes the validator scores (np.diff of the path on the interval grid), not on prices, so
the diagnosis lines up one-to-one with what CRPS charges us for.

TSLAX is included as the within-class control: same code, same window, +0.6% instead of +11.3%. If
NVDAX's PIT is deformed in a way TSLAX's is not, that deformation IS the deficit. If both look the
same, then our marginals are fine and the leaders are winning with information we do not have --
which is a publishable negative result and the point at which to stop spending on this competition.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='NVDAX,GOOGLX,TSLAX,AAPLX')
ap.add_argument('--slug', default='com-equ-24h')
ap.add_argument('--start', default='2026-07-16')
ap.add_argument('--end', default='2026-07-26')
ap.add_argument('--every-min', type=int, default=180)
ap.add_argument('--nsim', type=int, default=400)
# the intervals com-equ-24h actually scores, in 5-min steps
ap.add_argument('--intervals', default='')
a = ap.parse_args()

# geometry comes from the competition config, not hardcoded, so this runs on crypto-1h
# (1-min steps, 60-min horizon) as well as the two 24h competitions (5-min steps).
_comp = harness.COMPS[a.slug]
STEP = _comp.time_increment / 60.0          # minutes per step
HORIZON = _comp.time_length / 60.0          # minutes of horizon
_DEFAULT_IV = {300: '1,6,12,36,288', 60: '1,2,5,12,60'}
ivs = [int(x) for x in (a.intervals or _DEFAULT_IV[_comp.time_increment]).split(',')]

print(f"PIT of interval changes, {a.assets}  {a.start}->{a.end}  nsim={a.nsim}")
print("bins are quintiles of u = P(ensemble <= realized); uniform = 20% each\n")
print(f"{'asset':8s}{'iv(min)':>9s}{'n':>6s}" + "".join(f"{'q'+str(i+1):>8s}" for i in range(5))
      + f"{'outside':>9s}{'shape':>12s}")

starts = pd.date_range(a.start, a.end, freq=f'{a.every_min}min', tz='UTC')

for asset in a.assets.split(','):
    acc = {iv: [] for iv in ivs}
    for t in starts:
        try:
            ctx = harness.context(asset, t)
            if ctx is None or len(ctx) < 2000:
                continue
            res = modeling.simulate(asset=asset, start_time=t, time_increment=int(STEP * 60),
                                    time_length=int(HORIZON * 60), num_simulations=a.nsim,
                                    context_prices=ctx)
            paths = np.asarray(res[2:], dtype=float)
            real = harness.realized(asset, t, int(HORIZON * 60), int(STEP * 60))
            if real is None or len(real) != paths.shape[1] or not np.isfinite(real).all():
                continue
        except Exception:
            continue
        for iv in ivs:
            # non-overlapping interval changes, exactly as the validator forms them
            pe = paths[:, ::iv]
            re_ = np.asarray(real)[::iv]
            if pe.shape[1] < 2:
                continue
            de = np.diff(pe, axis=1) / pe[:, :-1]
            dr = np.diff(re_) / re_[:-1]
            for j in range(de.shape[1]):
                col = de[:, j]
                if not np.isfinite(dr[j]) or not np.isfinite(col).all():
                    continue
                acc[iv].append(float((col < dr[j]).mean()))
    for iv in ivs:
        u = np.asarray(acc[iv])
        if len(u) < 50:
            continue
        h = np.histogram(u, bins=5, range=(0, 1))[0] / len(u) * 100
        outside = float(((u <= 0.005) | (u >= 0.995)).mean() * 100)
        edge, mid = h[0] + h[4], h[2]
        if edge > 48:
            shape = "TOO NARROW"
        elif mid > 26:
            shape = "TOO WIDE"
        elif abs(h[0] - h[4]) > 9:
            shape = "TILTED"
        else:
            shape = "ok"
        print(f"{asset:8s}{int(iv*STEP):9d}{len(u):6d}" + "".join(f"{x:7.1f}%" for x in h)
              + f"{outside:8.1f}%{shape:>12s}")
    print()

print("'outside' = fraction of realizations beyond the ensemble's full support (u<=0.5% or >=99.5%).")
print("CRPS punishes those hardest, so an elevated 'outside' at the high-weight intervals is the")
print("single most actionable defect a PIT can reveal.")
