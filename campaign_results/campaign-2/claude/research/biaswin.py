"""Is the +25% sigma bias SYSTEMATIC or an artifact of a quiet evaluation window?

This is the decisive question before touching `scale`. If realized/forecast is ~0.8 in every
window across months, the vol model has a genuine level bias and correcting it is real. If it is
0.8 only in late July because late July was quiet, then hard-coding 0.8 is overfitting to the
window right before the evaluation period -- and if vol mean-reverts up during evaluation, an
under-dispersed ensemble is punished hard (CRPS is much steeper on the under-dispersed side).

Runs the model's own forecast against realized on a rolling grid of prompts, straight from the
snapshot (no field data needed), so we can cover the full history cheaply.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/agent/research'); sys.path.insert(0, '/workspace/agent')
import harness
import importlib

ap = argparse.ArgumentParser()
ap.add_argument('--assets', default='BTC,ETH,SOL,XRP,HYPE')
ap.add_argument('--mod', default='modeling')
ap.add_argument('--start', default='2026-03-15')
ap.add_argument('--end', default='2026-07-27')
ap.add_argument('--every-min', type=int, default=360)
ap.add_argument('--horizon-min', type=int, default=60)
ap.add_argument('--step-min', type=float, default=1.0)
a = ap.parse_args()

mod = importlib.import_module(a.mod)
step_min = a.step_min
n_steps = int(a.horizon_min / step_min)

rows = []
for asset in a.assets.split(','):
    s = harness.load_prices(asset)
    season = mod._season_curve(asset)
    grid = pd.date_range(pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC'),
                         freq=f'{a.every_min}min')
    for t in grid:
        if t not in s.index:
            continue
        ctx = harness.context(asset, t)
        if len(ctx) < 2000:
            continue
        real = harness.realized(asset, t, int(a.horizon_min * 60), int(step_min * 60))
        if np.isnan(real).any():
            continue
        cr = np.diff(np.log(ctx.to_numpy()))
        idx = mod._hw_index(ctx.index[-1] - pd.Timedelta(minutes=len(cr)), len(cr), 1.0)
        cd = cr / np.maximum(season[idx], 1e-6)
        sig_min = mod._forecast_sigma(asset, cd)
        _, ratio = mod._pool_for(asset, step_min)
        fut = mod._hw_index(t, n_steps, step_min)
        amp = season[fut] ** float(mod.PARAMS['season_pow'])
        step_sig = sig_min * amp * step_min ** float(mod.PARAMS['hurst']) * ratio
        rr = np.diff(np.log(real))
        rows.append({'asset': asset, 't': t,
                     'f': float(np.sqrt(np.mean(step_sig ** 2))),
                     'r': float(np.sqrt(np.mean(rr ** 2)))})

df = pd.DataFrame(rows)
df['ratio'] = df.r / df.f
df['month'] = df.t.dt.to_period('M')
print(f"n={len(df)} prompts, horizon={a.horizon_min}min step={step_min}min\n")
print("=== realized/forecast sigma ratio BY MONTH (per asset) ===")
pt = df.pivot_table(index='month', columns='asset', values='ratio', aggfunc='median')
pt['ALL'] = df.groupby('month')['ratio'].median()
print(pt.round(3).to_string())
print("\n=== by half-month ===")
df['half'] = df.t.dt.strftime('%Y-%m-') + np.where(df.t.dt.day <= 15, 'a', 'b')
h = df.groupby('half')['ratio'].agg(['median', 'mean', 'count'])
print(h.round(3).to_string())
print("\n=== overall ===")
for asset, g in df.groupby('asset'):
    print(f"  {asset:7s} median={g.ratio.median():.3f} mean={g.ratio.mean():.3f} "
          f"iqr=[{g.ratio.quantile(.25):.3f},{g.ratio.quantile(.75):.3f}] n={len(g)}")
print(f"  {'ALL':7s} median={df.ratio.median():.3f} mean={df.ratio.mean():.3f}")
