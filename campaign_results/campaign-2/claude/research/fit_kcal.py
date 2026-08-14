"""Fit the per-(asset, increment) CRPS-optimal dispersion multiplier `kcal` OUT OF SAMPLE.

Discipline: kcal is fitted ONLY on the training window (default 2026-07-08 -> 2026-07-20 for the
24h competitions, 2026-07-10 -> 2026-07-22 for crypto-1h) and every reported number comes from a
disjoint later window. That is the whole point -- a scale fitted on the reporting window would
be indistinguishable from overfitting the two days before the evaluation period.

Method: for each prompt, simulate ONCE and rescale the log-increments by each candidate k (same
shocks, pure dispersion change), score with the validator's CRPS, and pick the k minimising mean
CRPS. A parabola is fitted through the best three points for a sub-grid optimum, then the result
is shrunk toward 1.0 to avoid chasing sampling noise.
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


def fit_asset(mod, slug, asset, start, end, stride, nsim, ks, seed0=777):
    comp = harness.COMPS[slug]
    fp = harness.field_prompts(slug, asset)
    fp = fp[(fp.start_time >= pd.Timestamp(start, tz='UTC')) & (fp.start_time < pd.Timestamp(end, tz='UTC'))]
    fp = fp.iloc[::stride]
    tot = np.zeros(len(ks)); n = 0
    for i, t in enumerate(fp.start_time):
        ctx = harness.context(asset, t)
        if len(ctx) < 60:
            continue
        real = harness.realized(asset, t, comp.time_length, comp.time_increment)
        if not np.isfinite(real[0]):
            continue
        np.random.seed(seed0 + i)
        out = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=comp.time_increment,
                           time_length=comp.time_length, num_simulations=nsim, context_prices=ctx)
        p = np.asarray(out[2:], dtype=float)
        base = p[:, 0:1]
        lg = np.log(p / base)
        for j, k in enumerate(ks):
            c, _ = calculate_crps_for_miner(base * np.exp(lg * k), real,
                                            comp.time_increment, comp.scoring_intervals)
            tot[j] += c
        n += 1
    if n == 0:
        return None
    m = tot / n
    j = int(np.argmin(m))
    kbest = ks[j]
    # parabolic refine through the three points around the grid minimum
    if 0 < j < len(ks) - 1:
        x = np.array(ks[j - 1:j + 2]); y = m[j - 1:j + 2]
        d = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        if abs(d) > 1e-12:
            a2 = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / d
            a1 = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / d
            if a2 > 0:
                kbest = float(np.clip(-a1 / (2 * a2), ks[0], ks[-1]))
    return kbest, m, n


ap = argparse.ArgumentParser()
ap.add_argument('--slug', default='crypto-24h')
ap.add_argument('--assets', default='')
ap.add_argument('--train', default='2026-07-08:2026-07-20')
ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--nsim', type=int, default=400)
ap.add_argument('--ks', default='0.75,0.85,0.95,1.05,1.15,1.30')
ap.add_argument('--shrink', type=float, default=0.7, help='weight on the fitted k vs 1.0')
ap.add_argument('--write', action='store_true')
a = ap.parse_args()

mod = importlib.import_module('modeling')
mod.PARAMS['use_kcal'] = 0.0          # fit on the RAW forecast, never on top of an existing kcal
comp = harness.COMPS[a.slug]
inc_key = 5 if comp.time_increment >= 180 else 1
assets = a.assets.split(',') if a.assets else [x for x in comp.asset_list if x != 'SPYX']
ks = [float(x) for x in a.ks.split(',')]
s, e = a.train.split(':')

print(f"fitting kcal{inc_key} on {a.slug} train={s}->{e} stride={a.stride} nsim={a.nsim}")
print(f"{'asset':8s} {'n':>4s}" + "".join(f"{k:>9.2f}" for k in ks) + f"{'kraw':>8s}{'kshr':>8s}")
out = {}
for asset in assets:
    try:
        r = fit_asset(mod, a.slug, asset, s, e, a.stride, a.nsim, ks)
    except FileNotFoundError:
        print(f"{asset:8s} no bundle"); continue
    if r is None:
        print(f"{asset:8s} no usable prompts"); continue
    kbest, m, n = r
    kshr = 1.0 + a.shrink * (kbest - 1.0)
    print(f"{asset:8s} {n:4d}" + "".join(f"{x:9.1f}" for x in m) + f"{kbest:8.3f}{kshr:8.3f}")
    out[asset] = kshr

if a.write and out:
    z = dict(np.load(ART, allow_pickle=False))
    for asset, k in out.items():
        z[f'{asset}_kcal{inc_key}'] = np.array([k], dtype=float)
    np.savez_compressed(ART, **z)
    print(f"\nwrote kcal{inc_key} for {len(out)} assets into {ART}")
else:
    print("\n(dry run; pass --write to persist)")
