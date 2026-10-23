"""Diagnostic (not the metric): realised step returns standardised by the model's forecast sigma."""
import sys, os, importlib.util, argparse
import numpy as np, pandas as pd
sys.path.insert(0, '/workspace/agent')
from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices
ap = argparse.ArgumentParser(); ap.add_argument('--assets', nargs='+'); ap.add_argument('--start'); ap.add_argument('--end')
ap.add_argument('--ti', type=int, default=300); ap.add_argument('--tl', type=int, default=86400); ap.add_argument('--cad', type=int, default=67)
ap.add_argument('--model', default='/workspace/agent/modeling.py')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('m', a.model); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows = []
for asset in a.assets:
    s = pd.Timestamp(a.start, tz='UTC'); e = pd.Timestamp(a.end, tz='UTC')
    pr = load_minute_prices(__import__('pathlib').Path('/workspace/market_data'), asset, s - pd.Timedelta(days=8), min(e + pd.Timedelta(days=1), pd.Timestamp('2026-10-11 23:59', tz='UTC')))
    times = pd.date_range(s, e, freq=f'{a.cad}min')
    for t in times:
        ctx = pr.loc[:context_end(t, a.tl)][-10080:]
        sig, last, _ = m.forecast(asset, t.isoformat(), a.ti, a.tl, ctx)
        grid = t + pd.to_timedelta(np.arange(sig.size + 1) * a.ti, unit='s')
        # realised price at grid point g = close of bar labelled g-60s
        px = pr['close'].reindex(grid - pd.Timedelta(seconds=60)).to_numpy().copy()
        px[0] = last
        r = np.diff(np.log(px))
        ny = grid[1:].tz_convert('America/New_York')
        rows.append(pd.DataFrame({'asset': asset, 'z': r / sig, 'h': ny.hour, 'we': ny.dayofweek >= 5, 'k': np.arange(sig.size)}))
d = pd.concat(rows).dropna()
if os.environ.get('DUMP'): d.to_parquet(os.environ['DUMP'])
d = d[d.z != 0] if os.environ.get('NOZERO') else d
def summ(g):
    return pd.Series({'n': len(g), 'rms': np.sqrt((g.z**2).mean()), 'mad': (g.z.abs().median() / 0.6745), 'zero%': (g.z == 0).mean() * 100})
print(d.groupby('asset').apply(summ).round(3)); print(d.groupby(['asset','we']).apply(summ).round(3).unstack(1).to_string())
if len(a.assets) == 1:
    print(d.groupby(['we', 'h']).apply(summ).round(2).unstack(0).to_string())
