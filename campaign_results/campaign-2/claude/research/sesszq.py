"""Is the ZERO-ATOM rate session-dependent in a way the current state-conditioning already captures?

Report section 9 item 5 says: session-conditioning the innovation SHAPE was worth 0.33-0.47%, and the
same argument applies untested to the other session-blind components. The zero atom is the sharpest
of these, because session.py measured P(exact zero) ranging from 2.6% (US open) to 33.6% (weekend) --
a 13x swing, far larger than the kurtosis swing that made the shape fix worthwhile.

But there is a strong reason it might already be handled, and it is worth stating before spending
anything. _zero_prob does NOT use a static rate: it blends the asset's unconditional rate with the
EMPIRICAL rate over the trailing 12 steps. Staleness is highly persistent -- if the feed has been
frozen for the last hour it is probably still frozen -- so a trailing-window estimator may already
track the session implicitly, and better than a session label could, because it also catches
holidays, halts and outages that no fixed calendar knows about.

That makes this a genuine question rather than an obvious win, and it decides where the next unit of
effort goes. The test:

  1. Measure the realized zero rate per session, per asset (the target).
  2. Measure what the CURRENT trailing-12 estimator predicts at those same times.
  3. Compare. If the estimator already tracks the session pattern, a session-conditional zero rate is
     redundant and I should say so rather than build it. If it lags systematically -- in particular
     if it under-predicts at session BOUNDARIES, where the trailing window is drawn from the previous
     session -- then the residual is exactly what a session term would add, and its size tells me
     whether it is worth having.

The boundary case is the one that matters for scoring: a 24h path issued at 20:00 UTC on a Friday has
a trailing window full of active open-market minutes and a horizon that is almost entirely weekend.
That is precisely where a trailing estimator is worst and a calendar is best, and it is also a large
share of the com-equ prompts.

Note this only measures the ATOM RATE, not whether modelling the atom helps -- iteration 8 already
refuted zero-inflation as implemented (zi_mix is 0). So a positive result here does not resurrect it
on its own; it would only justify revisiting it with a correct rate. Reported honestly either way.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='AAPLX,XAU,NVDAX,TSLAX,GOOGLX,WTIOIL')
ap.add_argument('--step', type=int, default=5, help='increment in minutes (5 = the 24h competitions)')
ap.add_argument('--end', default='2026-07-27')
a = ap.parse_args()


def session_of(idx):
    dow = idx.dayofweek.to_numpy()
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    s = np.full(len(idx), 'closed', dtype=object)
    s[dow >= 5] = 'weekend'
    wd = dow < 5
    s[wd & (hod >= 13.5) & (hod < 20.0)] = 'open'
    s[wd & (((hod >= 9.5) & (hod < 13.5)) | ((hod >= 20.0) & (hod < 24.0)))] = 'extended'
    return s


print(f"zero-atom rate: realized vs what the trailing-12 estimator predicts (step={a.step}min)")
print("'boundary' = steps whose trailing window sits mostly in a DIFFERENT session\n")
print(f"{'asset':8s}{'session':10s}{'n':>7s}{'realized':>10s}{'trail-12':>10s}{'err':>8s}"
      f"{'bnd real':>10s}{'bnd trail':>11s}{'bnd err':>9s}")

rows = []
for asset in a.assets.split(','):
    try:
        s = harness.load_prices(asset)
    except Exception as e:
        print(f"{asset:8s} {e}"); continue
    s = s[np.isfinite(s.values)]
    s = s[s.index <= pd.Timestamp(a.end, tz='UTC')]
    si = s[::a.step]
    r = pd.Series(np.diff(np.log(si.values)), index=si.index[1:])
    r = r[np.isfinite(r)]
    isz = (r.values == 0.0).astype(float)
    # the estimator modeling._zero_prob actually uses: mean of the trailing 12 steps, causal
    trail = pd.Series(isz, index=r.index).rolling(12).mean().shift(1).to_numpy()
    sess = session_of(r.index)
    prev = np.roll(sess, 12)              # session 12 steps ago
    boundary = (prev != sess)
    ok = np.isfinite(trail)
    for name in ['open', 'extended', 'closed', 'weekend']:
        m = (sess == name) & ok
        if m.sum() < 200:
            continue
        real, pred = isz[m].mean() * 100, np.nanmean(trail[m]) * 100
        mb = m & boundary
        if mb.sum() >= 50:
            rb, pb = isz[mb].mean() * 100, np.nanmean(trail[mb]) * 100
            bstr = f"{rb:9.1f}%{pb:10.1f}%{rb - pb:+8.1f}"
        else:
            bstr = f"{'-':>10s}{'-':>11s}{'-':>9s}"
        print(f"{asset:8s}{name:10s}{int(m.sum()):7d}{real:9.1f}%{pred:9.1f}%{real - pred:+8.1f}{bstr}")
        rows.append(dict(asset=asset, sess=name, err=real - pred))
    print()

if rows:
    d = pd.DataFrame(rows)
    print("=" * 78)
    print("mean estimator error by session (realized - predicted, percentage points):")
    print(d.groupby('sess').err.mean().round(2).to_string())
    mx = d.groupby('sess').err.mean().abs().max()
    print(f"\nlargest systematic session bias: {mx:.2f} pp")
    print("  < ~2pp  -> the trailing estimator already tracks the session; a session-conditional")
    print("             zero rate is redundant. Say so and do not build it.")
    print("  > ~5pp  -> a real residual a calendar term would capture; worth revisiting the atom")
    print("             WITH a corrected rate (the atom itself was refuted in iteration 8).")
