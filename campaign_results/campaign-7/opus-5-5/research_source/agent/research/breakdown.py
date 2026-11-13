"""Per-interval CRPS breakdown (synth's own calculate_total_score_for_miner) on a sample of prompts."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.backtester.config import competition_for
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module

ap = argparse.ArgumentParser()
ap.add_argument('--asset', default='BTC'); ap.add_argument('--tl', type=int, default=86400)
ap.add_argument('--start', default='2026-10-01'); ap.add_argument('--end', default='2026-10-15')
ap.add_argument('--every', type=int, default=180, help='minutes between sampled prompts')
ap.add_argument('--model', default='agent/modeling.py'); ap.add_argument('--params', default=None)
a = ap.parse_args()
comp = competition_for(a.asset, a.tl); ti = comp.time_increment
mod = load_module(Path(a.model))
if a.params: mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod)
s, e = pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC')
prices = load_minute_prices(WS / 'market_data', a.asset, s - pd.Timedelta(days=cd), e + pd.Timedelta(seconds=a.tl))
rows = []
for t in pd.date_range(s, e, freq=f'{a.every}min', inclusive='left'):
    ctx = context_slice(prices, t, a.tl, cd)
    out = mod.simulate(asset=a.asset, start_time=t.isoformat(), time_increment=ti, time_length=a.tl, num_simulations=1000, context_prices=ctx)
    sims = np.asarray(out[2:], dtype=np.float32).astype(float)
    real = prices['close'].loc[t: t + pd.Timedelta(seconds=a.tl): ti // 60].to_numpy(dtype=float)
    tot, det = calculate_total_score_for_miner(sims, real, ti, comp.scoring_intervals, comp.vol_scoring_blocks)
    r = {d['Interval']: d['CRPS'] * d.get('Weight', 1.0) for d in det}
    r['t'] = t; rows.append(r)
df = pd.DataFrame(rows).set_index('t')
m = df.mean()
print(a.asset, a.tl, 'n=', len(df))
print((m.round(2)).to_string())
print('share of Overall:', (m / m['Overall'] * 100).round(1).drop('Overall').to_dict())
