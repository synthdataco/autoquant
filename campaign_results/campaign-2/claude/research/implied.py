"""What sigma does the FIELD's CRPS imply, and what sigma is optimal?

Legitimate under the rules: uses only the aggregate published CRPS numbers from the bundle plus
the snapshot prices. No miner's predictions are fetched or dissected.

Method. For a Gaussian ensemble with sd s against realized x, CRPS has a closed form:
    CRPS(s, x) = s * [ u*(2*Phi(u)-1) + 2*phi(u) - 1/sqrt(pi) ],  u = x/s
So for each prompt we can ask: what single scale multiplier k on OUR forecast sigma would have
minimised the realized CRPS? Averaged over prompts that reveals whether we are systematically
over- or under-dispersed -- the dominant CRPS error mode -- independent of tail shape.

We do this on the actual scored quantity (relative changes x 1e4 per interval) by rescaling our
own simulated increments, which is exactly the `scale` knob in modeling.PARAMS.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
from synth.validator.crps_calculation import calculate_crps_for_miner

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-1h')
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--window', default='2026-07-24:2026-07-26')
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--stride', type=int, default=6)
ap.add_argument('--mod', default='modeling')
ap.add_argument('--ks', default='0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.20')
a = ap.parse_args()

mod = importlib.import_module(a.mod)
comp = harness.COMPS[a.slug]
ks = [float(x) for x in a.ks.split(',')]
s, e = a.window.split(':')

print(f"scale sweep on the SAME simulated shocks (isolates dispersion from shape)")
print(f"{'asset':7s}" + "".join(f"{k:>9.2f}" for k in ks) + f"{'argmin':>9s}")
allrows = []
for asset in a.assets.split(','):
    fp = harness.field_prompts(a.slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(s, tz='UTC')) & (fp.start_time < pd.Timestamp(e, tz='UTC'))]
    fp = fp.iloc[::a.stride]
    tot = np.zeros(len(ks))
    n = 0
    for i, t in enumerate(fp.start_time):
        ctx = harness.context(asset, t)
        if len(ctx) < 60:
            continue
        real = harness.realized(asset, t, comp.time_length, comp.time_increment)
        if not np.isfinite(real[0]):
            continue
        np.random.seed(10_000 + i)
        out = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=comp.time_increment,
                           time_length=comp.time_length, num_simulations=a.nsim, context_prices=ctx)
        p = np.asarray(out[2:], dtype=float)
        last = p[:, 0:1]
        # rescale log-increments around the common start: pure dispersion change, same shocks
        lg = np.log(p / last)
        for j, k in enumerate(ks):
            pk = last * np.exp(lg * k)
            c, _ = calculate_crps_for_miner(pk, real, comp.time_increment, comp.scoring_intervals)
            tot[j] += c
        n += 1
    if n == 0:
        continue
    m = tot / n
    print(f"{asset:7s}" + "".join(f"{x:9.2f}" for x in m) + f"{ks[int(np.argmin(m))]:9.2f}")
    allrows.append(m)
if allrows:
    M = np.mean(allrows, axis=0)
    print(f"{'MEAN':7s}" + "".join(f"{x:9.2f}" for x in M) + f"{ks[int(np.argmin(M))]:9.2f}")
    b = M.min()
    print(f"{'rel%':7s}" + "".join(f"{(x/b-1)*100:8.2f}%" for x in M))
