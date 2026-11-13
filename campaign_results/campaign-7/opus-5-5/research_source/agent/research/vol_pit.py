"""PIT of the realised 60-min (and 15-min) volatility (std of 1-min returns, the 1h vol term's
statistic) within the simulated paths' distribution of the same statistic."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
mod = load_module(Path('agent/modeling.py'))
mod.PARAMS.update(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
cd = context_days_of(mod)
rng = np.random.default_rng(0)
out = {60: [], 15: []}
for asset in sys.argv[1].split(','):
    s, e = pd.Timestamp('2026-10-08', tz='UTC'), pd.Timestamp('2026-10-18', tz='UTC')
    prices = load_minute_prices(WS / 'market_data', asset, s - pd.Timedelta(days=cd), e + pd.Timedelta(hours=1))
    for t in pd.date_range(s, e, freq='70min', inclusive='left'):
        ctx = context_slice(prices, t, 3600, cd)
        o = mod.simulate(asset=asset, start_time=t.isoformat(), time_increment=60, time_length=3600, num_simulations=1000, context_prices=ctx)
        sims = np.asarray(o[2:], dtype=np.float32).astype(float)
        real = prices['close'].loc[t: t + pd.Timedelta(hours=1)].to_numpy(dtype=float)
        if not np.all(np.isfinite(real)):
            continue
        rs = np.diff(sims, axis=1) / sims[:, :-1] * 1e4
        rr = np.diff(real) / real[:-1] * 1e4
        for b in (60, 15):
            for j in range(0, 60, b):
                vs = rs[:, j:j + b].std(axis=1, ddof=1); vr = rr[j:j + b].std(ddof=1)
                lo, hi = np.mean(vs < vr), np.mean(vs <= vr)
                out[b].append(lo + rng.random() * (hi - lo))
for b in (60, 15):
    h, _ = np.histogram(out[b], bins=10, range=(0, 1))
    print(f"vol block {b:2d} min n={len(out[b])} PIT% " + ' '.join(f'{100 * x / len(out[b]):4.1f}' for x in h))
