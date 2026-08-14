"""Does the MA(1) theta -- which controls ONLY autocovariance -- sit at the right value for the
anchored intervals that carry 27% of crypto-1h?

Chain of evidence that led here:
  ivdecomp.py  crypto-1h CRPS splits 1min 27.7%, Gaps 27.4%, 2min 19.3%, 5min 12.0%, rest 13.7%.
               The Gaps block is twelve anchored start->X intervals, X = 5..60 min.
  gappit.py    The anchored PIT is near-uniform at most horizons but TOO NARROW at the shortest
               anchored horizons (5-10min) on ETH, SOL and HYPE -- the ensemble's anchored spread
               is too small there.

theta is the right lever for precisely this and nothing else. The filter in modeling.simulate is
variance-normalized (divided by sqrt(1+th^2)), so changing theta leaves the ONE-STEP variance exactly
where kcal calibrated it and changes only the autocovariance between steps. Since

    Var(anchored over X) = X*Var(step) + 2*sum_{k<X} (X-k)*Cov(step_t, step_{t+k})

theta moves the anchored spread while holding the 1-min marginal fixed. The sign is unambiguous in
the code: out = ext - th*lag, so POSITIVE theta imposes negative autocorrelation and SHRINKS the
anchored variance. Currently ma1_1 = 0.10. An anchored distribution that is too narrow at short X
therefore points to a theta that is too high.

That is a clean directional prediction made BEFORE running the sweep, which is what makes this a test
rather than a fishing expedition: if the anchored distribution is genuinely too narrow, CRPS should
improve as theta falls below 0.10, and the improvement should come disproportionately from the Gaps
block rather than from 1min.

Every number is the validator's own CRPS through harness.evaluate -- no reimplemented scoring.

FIT/HOLDOUT IS MANDATORY. Iteration 17's kcal refit looked good on its fitting window and failed to
replicate; kcal_equ.py later rejected GOOGLX and WTIOIL on exactly this test. A theta that improves
the fit window and not the holdout is window-fitting and gets discarded, however good the fit looks.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import modeling
from harness import evaluate

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--key', default='ma1_1', help='PARAMS key to sweep (ma1_1 for 1min, ma1_5 for 5min)')
ap.add_argument('--vals', default='-0.10,-0.05,0.0,0.05,0.10,0.15')
ap.add_argument('--fit', default='2026-07-11:2026-07-18')
ap.add_argument('--holdout', default='2026-07-18:2026-07-26')
ap.add_argument('--nsim', type=int, default=300)
ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--workers', type=int, default=7)
a = ap.parse_args()

vals = [float(x) for x in a.vals.split(',')]
cur = float(modeling.PARAMS[a.key])
print(f"sweeping PARAMS['{a.key}'] (current {cur}) on {a.slug}, nsim={a.nsim}, stride={a.stride}")
print("theta controls ONLY autocovariance -- the one-step variance is renormalized away.")
print("prediction from gappit: anchored too narrow at short X => lower theta should help.\n")


def run(window, v):
    s, e = window.split(':')
    tot = []
    for asset in a.assets.split(','):
        try:
            r = evaluate(modeling.simulate, a.slug, asset, s, e, nsim=a.nsim,
                         workers=a.workers, stride=a.stride, params={a.key: v})
            if len(r):
                tot.append((asset, float(r.crps.mean())))
        except Exception as ex:
            print(f"    {asset} {a.key}={v}: {type(ex).__name__} {ex}")
    return dict(tot)


base_fit = run(a.fit, cur)
base_hold = run(a.holdout, cur)

print(f"{'theta':>8s}{'fit %':>10s}{'holdout %':>12s}{'replicates':>12s}")
best = None
for v in vals:
    f, h = run(a.fit, v), run(a.holdout, v)
    fd = np.mean([(f[k] / base_fit[k] - 1) * 100 for k in f if k in base_fit and base_fit[k] > 0])
    hd = np.mean([(h[k] / base_hold[k] - 1) * 100 for k in h if k in base_hold and base_hold[k] > 0])
    rep = "yes" if (fd < -0.02 and hd < -0.02) else "no"
    star = " *" if v == cur else ""
    print(f"{v:8.3f}{fd:+9.3f}%{hd:+11.3f}%{rep:>12s}{star}")
    if rep == "yes" and (best is None or fd + hd < best[1]):
        best = (v, fd + hd)

print("\n(negative = better CRPS than the current theta; both columns must improve to count)")
if best:
    print(f"=> theta={best[0]} improves BOTH windows. Adopt only if the holdout gain is meaningful.")
else:
    print("=> no theta improves both windows: the current value stands, and the anchored")
    print("   narrowness is NOT reachable through autocovariance. Record as a null result.")
