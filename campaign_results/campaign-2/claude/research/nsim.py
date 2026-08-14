"""How much CRPS do we lose to finite-ensemble bias, and what does the champion cost in latency?

crps_ensemble with M samples is biased UPWARD by ~E|X-X'|/(2M): the empirical CDF of a finite
sample is rougher than the true one. So research numbers at M=400 overstate the CRPS of the same
model served at M=1000. Two consequences worth pinning down:
  * the correction to apply when comparing our research numbers to the field (who serve 1000);
  * whether we are leaving anything on the table (we must honour num_simulations, so we cannot
    serve more than asked -- but we can confirm 1000 is near-converged).
"""
import argparse
import importlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
from synth.validator.crps_calculation import calculate_crps_for_miner

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,SOL')
ap.add_argument('--window', default='2026-07-24:2026-07-26')
ap.add_argument('--stride', type=int, default=12)
ap.add_argument('--ms', default='100,250,400,1000,2000')
a = ap.parse_args()

mod = importlib.import_module('modeling')
comp = harness.COMPS[a.slug]
ms = [int(x) for x in a.ms.split(',')]
s, e = a.window.split(':')

print(f"{a.slug}  intervals={len(comp.scoring_intervals)}  steps={comp.time_length//comp.time_increment+1}")
print(f"{'asset':7s}" + "".join(f"{m:>10d}" for m in ms))
rows = []
tim = {m: [] for m in ms}
for asset in a.assets.split(','):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(s, tz='UTC')) & (fp.start_time < pd.Timestamp(e, tz='UTC'))]
    fp = fp.iloc[::a.stride]
    tot = np.zeros(len(ms)); n = 0
    for i, t in enumerate(fp.start_time):
        ctx = harness.context(asset, t)
        if len(ctx) < 60:
            continue
        real = harness.realized(asset, t, comp.time_length, comp.time_increment)
        if not np.isfinite(real[0]):
            continue
        for j, m in enumerate(ms):
            np.random.seed(31_000 + i)
            t0 = time.perf_counter()
            o = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=comp.time_increment,
                             time_length=comp.time_length, num_simulations=m, context_prices=ctx)
            tim[m].append(time.perf_counter() - t0)
            p = np.asarray(o[2:], dtype=float)
            c, _ = calculate_crps_for_miner(p, real, comp.time_increment, comp.scoring_intervals)
            tot[j] += c
        n += 1
    if n:
        mm = tot / n
        rows.append(mm)
        print(f"{asset:7s}" + "".join(f"{x:10.2f}" for x in mm))
if rows:
    M = np.mean(rows, axis=0)
    print(f"{'MEAN':7s}" + "".join(f"{x:10.2f}" for x in M))
    print(f"{'vs1000':7s}" + "".join(f"{(x/M[ms.index(1000)]-1)*100:9.2f}%" for x in M) if 1000 in ms else "")
print("\n=== simulate() latency (seconds), full generation incl. .tolist() ===")
for m in ms:
    v = np.array(tim[m])
    print(f"  M={m:5d}  mean={v.mean():.3f}s  p95={np.quantile(v,0.95):.3f}s  max={v.max():.3f}s")
