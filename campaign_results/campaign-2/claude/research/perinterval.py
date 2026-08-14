"""Is the CRPS-optimal dispersion the SAME at every scored horizon? Per-interval scale diagnosis.

WHERE THIS SITS. kcal fixed the dispersion LEVEL per (asset, increment), worth several percent.
oracle.py then showed the remaining per-prompt vol-forecast error is worth at most 0.34% -- so
predicting *when* vol is high is exhausted. What has never been tested is whether the scale is right
as a function of HORIZON within a single prompt.

The model builds every horizon from one per-minute sigma times step_min**hurst, with a single
empirical `ratio` correction per increment. But crypto-1h is scored at SEVEN different horizons
simultaneously -- 1min, 2min, 5min, 15min, 30min, 60min_abs, plus 12 cumulative *_gaps windows that
together are 28.5% of the score. If the optimal multiplier is, say, 0.97 at 1min but 1.04 at the
60-min gap, then one global kcal is splitting the difference and losing at BOTH ends, and no amount
of level recalibration can fix it -- it needs a horizon-dependent scale (which is what `hurst`
nominally controls, but hurst was only ever swept coarsely against the TOTAL score, where opposing
per-interval errors mask each other).

THE MEASUREMENT. Reuse the official run's prediction files (1000 paths each, exactly what was
served). Rescale the log-increments by k on the SAME shocks -- pure dispersion, no MC noise -- and
find the CRPS-optimal k SEPARATELY FOR EACH scoring interval, using the validator's own CRPS.

READING IT:
  * all intervals want k ~ 1  -> the horizon profile is already right; this channel is closed too.
  * a monotone pattern in k across horizon -> hurst/ratio is misspecified; directly fixable, and the
    gain is the sum of the per-interval penalties at k=1, which is a real number this script prints.
"""
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
from synth.validator.crps_calculation import calculate_crps_for_miner

GAPS = {f'0_{m}min_gaps': m * 60 for m in range(5, 65, 5)}
INTERVALS_1H = {**GAPS, '1min': 60, '2min': 120, '5min': 300, '15min': 900, '30min': 1800,
                '60min_abs': 3600}
INTERVALS_24H = {'5min': 300, '30min': 1800, '3hour': 10800, '24hour_abs': 86400}

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='ch1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--ks', default='0.90,0.94,0.97,1.0,1.03,1.06,1.10')
ap.add_argument('--max-prompts', type=int, default=240)
a = ap.parse_args()

ks = [float(x) for x in a.ks.split(',')]
files = sorted(glob.glob(f"/workspace/miner_outputs/{a.tag}/predictions/*.json"))
by_asset = {}
for f in files:
    d = json.load(open(f))
    by_asset.setdefault(d['asset'], []).append(d)

print(f"per-interval CRPS-optimal dispersion multiplier, tag={a.tag}")
print("(same shocks, pure rescaling; k<1 = we are OVER-dispersed at that horizon)\n")

agg = {}
for asset in a.assets.split(','):
    items = by_asset.get(asset, [])[: a.max_prompts]
    if not items:
        continue
    # per interval: total CRPS at each k
    tot = {}
    n = 0
    for d in items:
        t = pd.Timestamp(d['start_timestamp'])
        inc, tl = int(d['time_increment']), int(d['time_length'])
        paths = np.asarray(d['paths'], dtype=float)
        real = harness.realized(asset, t, tl, inc)
        if not np.isfinite(real[0]):
            continue
        iv = INTERVALS_1H if tl == 3600 else INTERVALS_24H
        p0 = paths[:, :1]
        lg = np.log(paths / p0)            # log path relative to start
        n += 1
        for k in ks:
            pk = p0 * np.exp(lg * k)       # rescaled dispersion, same shocks
            for name, sec in iv.items():
                c, _ = calculate_crps_for_miner(pk, real, inc, {name: sec})
                if c == -1:
                    continue
                tot.setdefault(name, {}).setdefault(k, 0.0)
                tot[name][k] += c
    if n < 10:
        print(f"{asset}: too few ({n})"); continue
    print(f"{asset}  ({n} prompts)")
    print(f"  {'interval':12s}" + "".join(f"{f'k={k:g}':>10s}" for k in ks) +
          f"{'best k':>8s}{'gain':>8s}{'share':>7s}")
    grand = sum(v[1.0] for v in tot.values() if 1.0 in v)
    for name in tot:
        v = tot[name]
        base = v[1.0]
        bk = min(v, key=v.get)
        rel = {k: (v[k] / base - 1) * 100 for k in ks}
        agg.setdefault(name, []).append((bk, (v[bk] / base - 1) * 100, base / grand))
        print(f"  {name:12s}" + "".join(f"{rel[k]:+9.2f}%" for k in ks) +
              f"{bk:8.2f}{(v[bk]/base-1)*100:+7.2f}%{100*base/grand:6.1f}%")
    # total achievable if each interval could be scaled independently
    tg = sum(min(v.values()) for v in tot.values())
    t0 = sum(v[1.0] for v in tot.values())
    print(f"  -> if each interval were scaled optimally: {(tg/t0-1)*100:+.2f}% of total CRPS\n")

if agg:
    print("MEAN across assets:")
    print(f"  {'interval':12s}{'best k':>8s}{'gain':>8s}{'share':>7s}")
    for name, v in agg.items():
        bk = np.mean([x[0] for x in v]); g = np.mean([x[1] for x in v])
        sh = np.mean([x[2] for x in v])
        print(f"  {name:12s}{bk:8.3f}{g:+7.2f}%{100*sh:6.1f}%")
