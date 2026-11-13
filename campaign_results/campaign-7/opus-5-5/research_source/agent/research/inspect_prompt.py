"""Inspect one prompt: model sigma per step vs realised returns, by 3h block; CRPS breakdown."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.backtester.config import competition_for
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
ap = argparse.ArgumentParser()
ap.add_argument('--asset'); ap.add_argument('--t'); ap.add_argument('--tl', type=int, default=86400)
ap.add_argument('--params', default=None)
a = ap.parse_args()
mod = load_module(Path('agent/modeling.py'))
if a.params: mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod); comp = competition_for(a.asset, a.tl); ti = comp.time_increment; sm = ti // 60
t = pd.Timestamp(a.t, tz='UTC')
prices = load_minute_prices(WS / 'market_data', a.asset, t - pd.Timedelta(days=cd), t + pd.Timedelta(seconds=a.tl))
ctx = context_slice(prices, t, a.tl, cd)
last, var, mow, _ = mod._forecast(a.asset, t.isoformat(), ti, a.tl, ctx)
real = prices['close'].loc[t: t + pd.Timedelta(seconds=a.tl): sm].to_numpy(dtype=float)
r = np.diff(np.log(real)) * 1e4
sig = np.sqrt(var) * 1e4
blk = len(r) // 8
print(f"{a.asset} {t} last ctx {last} real0 {real[0]}  ({(real[0]/last-1)*1e4:.1f} bps)")
for b in range(8):
    sl = slice(b * blk, (b + 1) * blk)
    rr = r[sl]
    print(f" block {b}: model sigma {sig[sl].mean():6.2f} bps | realised rms {np.sqrt(np.mean(rr**2)):6.2f} mean|r| {np.mean(np.abs(rr)):6.2f} zero% {100*np.mean(rr==0):4.0f} | cum {np.sum(rr):7.1f}")
# context recent behaviour
c = ctx['close'].dropna()
cr = np.diff(np.log(c.to_numpy()[-(6*60):][::sm])) * 1e4
print(f" context last 6h {sm}-min rms {np.sqrt(np.mean(cr**2)):.2f} bps, zero% {100*np.mean(cr==0):.0f}")
out = mod.simulate(asset=a.asset, start_time=t.isoformat(), time_increment=ti, time_length=a.tl, num_simulations=1000, context_prices=ctx)
sims = np.asarray(out[2:], dtype=np.float32).astype(float)
tot, det = calculate_total_score_for_miner(sims, real, ti, comp.scoring_intervals, comp.vol_scoring_blocks)
print(' CRPS', {d['Interval']: round(d['CRPS'], 1) for d in det})
