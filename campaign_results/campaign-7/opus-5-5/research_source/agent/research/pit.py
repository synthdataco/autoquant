"""PIT histogram (10 bins) of k-step returns under the simulated fan (randomised ties)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
ap = argparse.ArgumentParser()
ap.add_argument('--asset'); ap.add_argument('--tl', type=int, default=3600); ap.add_argument('--params', default='{}')
ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--every', type=int, default=60)
ap.add_argument('--ks', default='1,5,60')
a = ap.parse_args()
mod = load_module(Path('agent/modeling.py')); mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod); ti = 60 if a.tl == 3600 else 300; sm = ti // 60
s, e = pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC')
prices = load_minute_prices(WS / 'market_data', a.asset, s - pd.Timedelta(days=cd), e + pd.Timedelta(seconds=a.tl))
rng = np.random.default_rng(0)
ks = [int(x) for x in a.ks.split(',')]
pits = {k: [] for k in ks}
for t in pd.date_range(s, e, freq=f'{a.every}min', inclusive='left'):
    ctx = context_slice(prices, t, a.tl, cd)
    out = mod.simulate(asset=a.asset, start_time=t.isoformat(), time_increment=ti, time_length=a.tl, num_simulations=1000, context_prices=ctx)
    sims = np.log(np.asarray(out[2:], dtype=np.float64))
    real = np.log(prices['close'].loc[t: t + pd.Timedelta(seconds=a.tl): sm].to_numpy(dtype=float))
    for k in ks:
        sr = sims[:, k::k] - sims[:, :-k:k][:, :sims[:, k::k].shape[1]]
        rr = real[k::k] - real[:-k:k][:len(real[k::k])]
        for j in range(len(rr)):
            if not np.isfinite(rr[j]):
                continue
            lo = np.mean(sr[:, j] < rr[j]); hi = np.mean(sr[:, j] <= rr[j])
            pits[k].append(lo + rng.random() * (hi - lo))
for k in ks:
    h, _ = np.histogram(pits[k], bins=10, range=(0, 1))
    print(f"{a.asset} k={k:3d} n={len(pits[k]):6d} PIT% " + ' '.join(f'{100 * x / len(pits[k]):4.1f}' for x in h))
