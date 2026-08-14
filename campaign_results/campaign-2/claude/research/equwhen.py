"""WHERE in the window does the com-equ deficit live? Concentrated in a few prompts, or everywhere?

Setup. On 2026-07-16 -> 07-21 we are catastrophically behind on NVDAX (+11.9% vs the field min),
GOOGLX (+11.5%) and XAU (+8.1%), while TSLAX (+1.1%) and WTIOIL (+3.4%) are merely mediocre -- same
code, same asset class, same window. On the LATER window (07-21 -> 07-26) the same assets sit at
+3.5-4.1%. So the deficit is not a fixed property of the asset; it is switched on by something in
the first half of the window.

A global dispersion sweep has already ruled out the simplest explanation: scale 0.85 -> 1.08 moves
the mean by 0.8% against an 11.9% gap, and is nearly flat. So we are not merely too wide or too
narrow on average. That leaves two very different stories, which imply opposite fixes:

  (A) A FEW BAD PROMPTS. A handful of prompts where the model is wildly wrong (a jump, an earnings
      gap, a stale-feed episode) dominates the mean. CRPS is an average, so ~5 prompts at 10x cost
      out of 120 can move the mean by 10% on their own. The fix would be about tail/jump handling,
      and the MEDIAN prompt-level ratio would look fine.

  (B) A UNIFORM SHIFT. Every prompt in the early window is a bit worse. The fix would be about the
      state the model was in during those days -- most plausibly the volatility level, since a
      vol-regime change that our HAR forecast lags would hurt every prompt in the same direction.

The discriminator is the DISTRIBUTION of the per-prompt ratio ours/field_min, not its mean:
  median ~1.0 with a fat right tail  => (A), a few disasters
  median tracking the mean           => (B), a uniform shift

The script also breaks the ratio down by day and by weekday/weekend, because for a tokenized equity
the obvious candidate for a regime the model mishandles is the market-closed period: a 24h forecast
issued on Friday spans a weekend during which the underlying cash market never opens, and if our
seasonal amplitude does not narrow the fan enough for that, every Friday/Saturday prompt is too wide.
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
ap.add_argument('--assets', default='NVDAX,GOOGLX,XAU,TSLAX')
ap.add_argument('--start', default='2026-07-16')
ap.add_argument('--end', default='2026-07-21')
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--stride', type=int, default=1)
a = ap.parse_args()

for asset in a.assets.split(','):
    try:
        r = evaluate(modeling.simulate, a.slug, asset, a.start, a.end,
                     nsim=a.nsim, workers=13, stride=a.stride)
    except Exception as e:
        print(f"{asset}: FAILED {e}"); continue
    if len(r) == 0:
        print(f"{asset}: no prompts"); continue
    r = r.copy()
    r['ratio'] = r.crps / r.field_min
    t = pd.to_datetime(r.start_time, utc=True)
    r['dow'] = t.dt.dayofweek
    r['day'] = t.dt.strftime('%m-%d')
    r['hod'] = t.dt.hour

    q = r.ratio.quantile([0.5, 0.75, 0.9, 0.95, 1.0])
    print(f"\n===== {asset}  n={len(r)}  mean ratio={r.ratio.mean():.3f} =====")
    print(f"  median={q[0.5]:.3f}  p75={q[0.75]:.3f}  p90={q[0.9]:.3f}  "
          f"p95={q[0.95]:.3f}  max={q[1.0]:.3f}")
    # how much of the EXCESS over the field min comes from the worst 10% of prompts?
    exc = (r.crps - r.field_min)
    top10 = exc.nlargest(max(1, len(r) // 10)).sum()
    frac = top10 / exc.sum() * 100 if exc.sum() > 0 else np.nan
    print(f"  worst 10% of prompts carry {frac:.0f}% of the total excess "
          f"(uniform would be ~10-25%)")
    if q[0.5] < 1.02 and frac > 45:
        print("  => (A) a few disasters dominate; median prompt is fine")
    elif frac < 35:
        print("  => (B) uniform shift; every prompt is worse")
    else:
        print("  => mixed")

    print("  by day:")
    g = r.groupby('day').agg(n=('ratio', 'size'), ratio=('ratio', 'mean'))
    for d, row in g.iterrows():
        print(f"    {d}  n={int(row.n):3d}  ratio={row.ratio:.3f}")
    we = r[r.dow >= 4].ratio.mean()   # Fri+Sat+Sun starts span a closed market
    wd = r[r.dow < 4].ratio.mean()
    print(f"  Fri-Sun starts (span the closed market): {we:.3f}   Mon-Thu starts: {wd:.3f}")
    if np.isfinite(we) and np.isfinite(wd) and we - wd > 0.02:
        print("  => market-closed spans are the weak point: the fan is too wide when the")
        print("     underlying cash market never opens during the forecast horizon.")
