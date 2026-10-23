"""Diagnostic: split the official per-prompt score into its intervals (synth crps_calculation functions)."""
import sys, os, importlib.util, argparse
import numpy as np, pandas as pd
from pathlib import Path
from synth.validator.crps_calculation import calculate_total_score_for_miner
from synth_lib.backtester.config import competition_for
from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices
ap = argparse.ArgumentParser(); ap.add_argument('--assets', nargs='+'); ap.add_argument('--start'); ap.add_argument('--end')
ap.add_argument('--tl', type=int, default=3600); ap.add_argument('--cad', type=int, default=97)
ap.add_argument('--model', default='/workspace/agent/modeling.py')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('m', a.model); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows = []
for asset in a.assets:
    comp = competition_for(asset, a.tl); ti = comp.time_increment
    s = pd.Timestamp(a.start, tz='UTC'); e = pd.Timestamp(a.end, tz='UTC')
    pr = load_minute_prices(Path('/workspace/market_data'), asset, s - pd.Timedelta(days=8), min(e + pd.Timedelta(days=2), pd.Timestamp('2026-10-11 23:59', tz='UTC')))
    for t in pd.date_range(s, e, freq=f'{a.cad}min'):
        ctx = pr.loc[:context_end(t, a.tl)][-10080:]
        out = m.simulate(asset, t.isoformat(), ti, a.tl, 1000, ctx)
        sims = np.asarray(out[2:], dtype=np.float32).astype(float)
        grid = t + pd.to_timedelta(np.arange(a.tl // ti + 1) * ti, unit='s')
        real = pr['close'].reindex(grid - pd.Timedelta(seconds=60)).to_numpy().astype(float)
        if np.isnan(real).any(): continue
        tot, det = calculate_total_score_for_miner(sims, real, ti, comp.scoring_intervals, comp.vol_scoring_blocks)
        r = {'asset': asset, 'total': tot}
        for d in det:
            r[d['Interval']] = d['CRPS'] * d.get('Weight', 1.0)
        rows.append(r)
d = pd.DataFrame(rows)
print(d.groupby('asset').mean().round(2).to_string())
print((d.drop(columns='asset').mean() / d.total.mean()).round(3).to_string())
