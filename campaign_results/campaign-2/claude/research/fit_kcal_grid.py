"""Fit kcal for assets with no (or too few) field prompts in the training window.

SP500 replaced SPYX and only starts being prompted on 2026-07-23, so the field bundle gives it 59
prompts -- all inside the reporting window. But CRPS needs only our paths and the realized path,
both of which come from the snapshot, so we can fit on a SYNTHETIC prompt grid over SP500's own
earlier price history (available from 2026-03-18) and keep the fit out of sample.
"""
import argparse
import importlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
from synth.validator.crps_calculation import calculate_crps_for_miner

ART = '/workspace/agent/artifacts/calib.npz'

ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='com-equ-24h')
ap.add_argument('--assets', default='SP500')
ap.add_argument('--train', default='2026-05-01:2026-07-20')
ap.add_argument('--every-min', type=int, default=360)
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--ks', default='0.85,0.95,1.05,1.15,1.30,1.50')
ap.add_argument('--shrink', type=float, default=0.7)
ap.add_argument('--write', action='store_true')
a = ap.parse_args()

mod = importlib.import_module('modeling')
mod.PARAMS['use_kcal'] = 0.0
comp = harness.COMPS[a.slug]
inc_key = 5 if comp.time_increment >= 180 else 1
ks = [float(x) for x in a.ks.split(',')]
s, e = a.train.split(':')

print(f"fitting kcal{inc_key} on a SYNTHETIC grid, {a.slug}, train={s}->{e}, every {a.every_min}min")
print(f"{'asset':8s} {'n':>4s}" + "".join(f"{k:>9.2f}" for k in ks) + f"{'kraw':>8s}{'kshr':>8s}")
out = {}
for asset in a.assets.split(','):
    grid = pd.date_range(pd.Timestamp(s, tz='UTC'), pd.Timestamp(e, tz='UTC'), freq=f'{a.every_min}min')
    tot = np.zeros(len(ks)); n = 0
    for i, t in enumerate(grid):
        ctx = harness.context(asset, t)
        if len(ctx) < 2000:
            continue
        real = harness.realized(asset, t, comp.time_length, comp.time_increment)
        if not np.isfinite(real[0]) or np.isnan(real).mean() > 0.2:
            continue
        np.random.seed(555 + i)
        o = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=comp.time_increment,
                         time_length=comp.time_length, num_simulations=a.nsim, context_prices=ctx)
        p = np.asarray(o[2:], dtype=float)
        base = p[:, 0:1]; lg = np.log(p / base)
        for j, k in enumerate(ks):
            c, _ = calculate_crps_for_miner(base * np.exp(lg * k), real,
                                            comp.time_increment, comp.scoring_intervals)
            tot[j] += c
        n += 1
    if n == 0:
        print(f"{asset:8s} no usable windows"); continue
    m = tot / n
    j = int(np.argmin(m)); kbest = ks[j]
    if 0 < j < len(ks) - 1:
        x = np.array(ks[j - 1:j + 2]); y = m[j - 1:j + 2]
        d = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        if abs(d) > 1e-12:
            a2 = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / d
            a1 = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / d
            if a2 > 0:
                kbest = float(np.clip(-a1 / (2 * a2), ks[0], ks[-1]))
    kshr = 1.0 + a.shrink * (kbest - 1.0)
    print(f"{asset:8s} {n:4d}" + "".join(f"{x:9.1f}" for x in m) + f"{kbest:8.3f}{kshr:8.3f}")
    out[asset] = kshr

if a.write and out:
    z = dict(np.load(ART, allow_pickle=False))
    for asset, k in out.items():
        z[f'{asset}_kcal{inc_key}'] = np.array([k], dtype=float)
    np.savez_compressed(ART, **z)
    print(f"\nwrote kcal{inc_key} for {list(out)} into {ART}")
else:
    print("\n(dry run; pass --write to persist)")
