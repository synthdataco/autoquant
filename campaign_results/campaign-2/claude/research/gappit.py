"""PIT of the ANCHORED (gaps) return distribution -- the 27% of crypto-1h I never diagnosed.

ivdecomp.py measured where crypto-1h's CRPS is actually spent: 1min 27.7%, Gaps 27.4%, 2min 19.3%,
then 5/15/30min and 60min_abs. The `Gaps` block is twelve `0_Xmin_gaps` intervals, X = 5..60, and
each scores a SINGLE anchored change from the path's start to X minutes in. My PIT work has only
ever looked at non-overlapping increments, so more than a quarter of my biggest score term has never
been checked for calibration.

There is a specific reason to expect the anchored distribution to be wrong even if every increment
marginal is right. CRPS on increments is marginal-only: per-step shape is charged directly and no
cross-step structure can hide behind it -- which is why I have been able to reason step-by-step so
far. The anchored return breaks that. The anchored change over X steps is the SUM of X increments,
so its variance is

    Var(sum) = X * Var(step) + 2 * sum_{k<X} (X-k) * Cov(step_t, step_{t+k})

The model applies MA(1)/MA(2) filters to the increments (ma1/ma2 in PARAMS), which set exactly those
covariance terms. Those filters were tuned to fit the 1-min increment autocorrelation of the price
series -- never against the anchored distribution they also control. A small error in the filter is
invisible at 1min (it barely changes the step marginal) and is AMPLIFIED linearly in X at the
anchored horizons, where it is then charged twelve times over, in twelve nested and highly correlated
intervals. Errors that are charged twelve times and diversified zero times are exactly the kind worth
hunting.

The test mirrors pit.py but forms the increments the way the validator forms `_gaps` ones:
paths[:, ::steps][:, :2] then a relative change -- i.e. start to X minutes, nothing else. It reports,
per anchored horizon:

  the PIT quintiles and the 'outside' rate (as in pit.py), and
  RATIO = sd(realized anchored return) / mean sd(ensemble anchored return)

RATIO is the direct read on the amplification argument: >1 means our anchored spread is too NARROW,
<1 too wide. If RATIO drifts systematically with X -- fine at 5min, worsening out to 60min -- that is
the autocovariance signature, and it is fixable by the ma1/ma2 filter. If RATIO is flat and near 1,
the anchored distribution is simply inherited correctly from the increments, the gaps block is not a
defect, and I record that and stop. Either way this is the last unexamined structural channel in the
competition that carries most of my score.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import modeling

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='')
ap.add_argument('--start', default='2026-07-16')
ap.add_argument('--end', default='2026-07-26')
ap.add_argument('--every-min', type=int, default=90)
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--params', default='', help='k=v,k=v overrides passed to modeling.PARAMS')
a = ap.parse_args()

comp = harness.COMPS[a.slug]
assets = (a.assets or ','.join(comp.asset_list)).split(',')
TI, TL = comp.time_increment, comp.time_length

if a.params:
    for kv in a.params.split(','):
        k, v = kv.split('=')
        modeling.PARAMS[k] = float(v)
    print(f"PARAMS overrides: {a.params}")

# the twelve anchored horizons the competition actually scores, in seconds
gap_secs = [v for k, v in comp.scoring_intervals.items() if k.endswith('_gaps')]
gap_secs = sorted(set(gap_secs))

print(f"ANCHORED (gaps) PIT -- {a.slug}, {a.start}->{a.end}, nsim={a.nsim}")
print("anchored change = start -> X min, exactly as calculate_price_changes_over_intervals builds it\n")
print(f"{'asset':7s}{'X(min)':>7s}{'n':>6s}" + "".join(f"{'q'+str(i+1):>7s}" for i in range(5))
      + f"{'outside':>9s}{'RATIO':>8s}{'shape':>12s}")

rows = []
for asset in assets:
    acc = {g: [] for g in gap_secs}
    sds = {g: [] for g in gap_secs}
    rls = {g: [] for g in gap_secs}
    for t in pd.date_range(a.start, a.end, freq=f'{a.every_min}min', tz='UTC'):
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
        except Exception:
            continue
        real = np.asarray(real, dtype=float)
        for g in gap_secs:
            steps = g // TI
            pe = paths[:, ::steps][:, :2]
            re_ = real[::steps][:2]
            if pe.shape[1] < 2 or len(re_) < 2:
                continue
            de = (pe[:, 1] - pe[:, 0]) / pe[:, 0]
            dr = (re_[1] - re_[0]) / re_[0]
            if not np.isfinite(dr) or not np.isfinite(de).all():
                continue
            acc[g].append(float((de < dr).mean()))
            sds[g].append(float(de.std()))
            rls[g].append(float(dr))

    for g in gap_secs:
        u = np.asarray(acc[g])
        if len(u) < 30:
            continue
        h = np.histogram(u, bins=5, range=(0, 1))[0] / len(u) * 100
        outside = float(((u <= 0.005) | (u >= 0.995)).mean() * 100)
        # realized dispersion vs ensemble dispersion, on the same prompts
        ratio = float(np.std(rls[g]) / np.mean(sds[g])) if np.mean(sds[g]) > 0 else np.nan
        edge, mid = h[0] + h[4], h[2]
        shape = "TOO NARROW" if edge > 48 else "TOO WIDE" if mid > 26 else \
                "TILTED" if abs(h[0] - h[4]) > 9 else "ok"
        print(f"{asset:7s}{g // 60:7d}{len(u):6d}" + "".join(f"{x:6.1f}%" for x in h)
              + f"{outside:8.1f}%{ratio:8.3f}{shape:>12s}")
        rows.append(dict(asset=asset, X=g // 60, ratio=ratio, outside=outside))
    print()

if rows:
    d = pd.DataFrame(rows)
    print("=" * 74)
    print("RATIO = sd(realized anchored) / sd(ensemble anchored), averaged across assets:")
    print(d.groupby('X').ratio.mean().round(3).to_string())
    g = d.groupby('X').ratio.mean()
    print(f"\nRATIO at 5min = {g.iloc[0]:.3f}, at 60min = {g.iloc[-1]:.3f}, "
          f"drift = {g.iloc[-1] - g.iloc[0]:+.3f}")
    print("  drift ~ 0        -> anchored spread is inherited correctly from the increments; the")
    print("                      gaps block is not a defect. Record and stop.")
    print("  drift systematic -> the MA filter's autocovariance is mis-set; it is invisible at 1min")
    print("                      and amplified in X. Fixable via ma1/ma2, and worth 27% of crypto-1h.")
