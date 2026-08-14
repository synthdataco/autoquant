"""Are theta and the dispersion scale trading off along a ridge -- and is the current point ON it?

vratio.py produced the sharpest tension of the campaign. The realized variance-ratio curve and the
model's implied one agree on SHAPE (fraction of total decay completed by X=10: realized 0.94, model
0.92 -- an MA(1) is the right functional form) and disagree badly on LEVEL:

    X          1     2     3     5    10    20    30    60
    realized 1.000 0.972 0.969 0.963 0.938 0.936 0.942 0.934
    model    1.000 0.901 0.868 0.842 0.822 0.812 0.809 0.805

The variance-normalized MA(1) at theta=0.10 imposes lag-1 autocorrelation of -0.099; the price series
shows about -0.028. The filter is roughly 3.5x too negative, and the implied missing spread at the
anchored horizons is ~7%. gappit.py independently found the anchored distribution TOO NARROW at 5-10
min, which is the same defect seen from the other side. Two independent diagnostics agree.

And yet the CRPS sweeps say theta=0.10 (1min) and 0.20 (5min) are optimal, with LOWER theta -- which
would raise the anchored spread, exactly the correction both diagnostics ask for -- strictly worse on
fit and holdout (iterations 26, 29). Three measurements, one contradiction. That is the most
interesting position to be in, and it has one obvious candidate resolution:

    theta is not an estimate of autocorrelation. It is a SHAPE parameter fitted jointly with the
    dispersion calibration `kcal`, which was itself fitted with theta in place. Lowering theta raises
    spread at every horizon X >= 2 while leaving X = 1 fixed; if kcal has already absorbed the
    resulting level error by scaling everything down, then moving theta alone breaks a calibration
    that was jointly optimal, and CRPS correctly gets worse -- even though the autocovariance is, in
    isolation, mis-specified.

If that is right, the CRPS surface over (theta, scale) has a RIDGE, the current point sits on it, and
the one-dimensional sweeps I ran were cutting ACROSS the ridge rather than along it. The prize, if
any, is at a different point along the ridge -- lower theta AND a compensating scale -- which no
one-dimensional sweep could ever have found.

This sweeps the two jointly. Reading the result:

  the best cell is the current point (or ties it)
      -> the joint calibration is already optimal, the VR mismatch is absorbed rather than costly,
         and §9 item 1 -- the report's best-identified open lead -- is closed. The mis-specified
         autocovariance is real and harmless, which is a publishable finding in itself.
  a different cell wins on BOTH fit and holdout
      -> the one-dimensional sweeps were misleading, and there is a genuine gain that only a joint
         search reveals. That is worth adopting and worth writing up loudly.

Same fit/holdout discipline as everywhere else: fit-window-only wins are discarded on principle.
Every number is the validator's own CRPS via harness.evaluate.
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
ap.add_argument('--assets', default='BTC,ETH,SOL')
ap.add_argument('--key', default='ma1_1')
ap.add_argument('--thetas', default='0.0,0.05,0.10')
ap.add_argument('--scales', default='0.97,1.0,1.03')
ap.add_argument('--fit', default='2026-07-14:2026-07-20')
ap.add_argument('--holdout', default='2026-07-20:2026-07-26')
ap.add_argument('--nsim', type=int, default=300)
ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--workers', type=int, default=7)
a = ap.parse_args()

thetas = [float(x) for x in a.thetas.split(',')]
scales = [float(x) for x in a.scales.split(',')]
cur_th = float(modeling.PARAMS[a.key])
fs, fe = a.fit.split(':')
hs, he = a.holdout.split(':')

print(f"joint (theta, scale) CRPS surface on {a.slug}, current theta={cur_th}, scale=1.0")
print(f"fit {a.fit} / holdout {a.holdout}, nsim={a.nsim}, stride={a.stride}")
print("cells are % vs the CURRENT point; negative = better\n")


def mean_crps(window, th, sc):
    s, e = window.split(':')
    vals = []
    for asset in a.assets.split(','):
        try:
            r = evaluate(modeling.simulate, a.slug, asset, s, e, nsim=a.nsim,
                         workers=a.workers, stride=a.stride, params={a.key: th, "scale": sc})
            if len(r):
                vals.append((asset, float(r.crps.mean())))
        except Exception as ex:
            print(f"  {asset} th={th} sc={sc}: {type(ex).__name__} {ex}")
    return dict(vals)


base_f = mean_crps(a.fit, cur_th, 1.0)
base_h = mean_crps(a.holdout, cur_th, 1.0)


def pct(d, base):
    ks = [k for k in d if k in base and base[k] > 0]
    return np.mean([(d[k] / base[k] - 1) * 100 for k in ks]) if ks else np.nan


print("FIT window:")
print(f"{'theta\\scale':>12s}" + "".join(f"{s:>10.2f}" for s in scales))
fit_grid, hold_grid = {}, {}
for th in thetas:
    cells = []
    for sc in scales:
        fit_grid[(th, sc)] = pct(mean_crps(a.fit, th, sc), base_f)
        cells.append(fit_grid[(th, sc)])
    print(f"{th:12.3f}" + "".join(f"{c:+9.3f}%" for c in cells))

print("\nHOLDOUT window:")
print(f"{'theta\\scale':>12s}" + "".join(f"{s:>10.2f}" for s in scales))
for th in thetas:
    cells = []
    for sc in scales:
        hold_grid[(th, sc)] = pct(mean_crps(a.holdout, th, sc), base_h)
        cells.append(hold_grid[(th, sc)])
    print(f"{th:12.3f}" + "".join(f"{c:+9.3f}%" for c in cells))

best = min(fit_grid, key=lambda k: fit_grid[k] + hold_grid[k])
print(f"\ncurrent point: (theta={cur_th}, scale=1.0)")
print(f"best joint cell: theta={best[0]}, scale={best[1]}  "
      f"fit {fit_grid[best]:+.3f}%  holdout {hold_grid[best]:+.3f}%")
if fit_grid[best] < -0.03 and hold_grid[best] < -0.03:
    print("=> a genuine ridge point beats the current one on BOTH windows. Adopt.")
else:
    print("=> no joint cell beats the current point on both windows: the calibration is already")
    print("   on the ridge, and the variance-ratio mis-specification is absorbed, not costly.")
