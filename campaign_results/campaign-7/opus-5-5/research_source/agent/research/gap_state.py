"""Gap to field p10 vs model state: quintiles of (forecast/slow-level) and (realised/forecast)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
ap = argparse.ArgumentParser()
ap.add_argument('--res'); ap.add_argument('--tl', type=int, default=3600); ap.add_argument('--params', default='{}')
ap.add_argument('--assets', default='BTC,SOL')
a = ap.parse_args()
mod = load_module(Path('agent/modeling.py')); mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod); ti = 60 if a.tl == 3600 else 300; sm = ti // 60
d = pd.read_parquet(f'agent/research/results/{a.res}.parquet')
d = d[d.asset.isin(a.assets.split(','))]
rows = []
for asset, g in d.groupby('asset'):
    st = pd.to_datetime(g.start_time, utc=True)
    prices = load_minute_prices(WS / 'market_data', asset, st.min() - pd.Timedelta(days=cd), st.max() + pd.Timedelta(seconds=a.tl))
    lpx = np.log(prices['close'])
    for (_, row), t in zip(g.iterrows(), st):
        ctx = context_slice(prices, t, a.tl, cd)
        last, var, mow, _ = mod._forecast(asset, t.isoformat(), ti, a.tl, ctx)
        real = lpx.loc[t: t + pd.Timedelta(seconds=a.tl): sm].to_numpy()
        r = np.diff(real)
        c = np.log(ctx['close'].dropna().to_numpy())
        rec = np.sqrt(np.mean(np.diff(c[-31:]) ** 2) * sm) if len(c) > 31 else np.nan
        rows.append({'asset': asset, 'gap': row.crps / row.p10 - 1, 'crps': row.crps, 'p10': row.p10,
                     'f': np.sqrt(np.mean(var)), 'rv': np.sqrt(np.nanmean(r ** 2)), 'rec30': rec})
x = pd.DataFrame(rows)
x['real_over_fc'] = x.rv / x.f
x['rec_over_fc'] = x.rec30 / x.f
f = lambda y: pd.Series({'n': len(y), 'gap%': 100 * (y.crps.sum() / y.p10.sum() - 1)})
for col in ['real_over_fc', 'rec_over_fc']:
    q = pd.qcut(x[col], 5)
    print(col, x.groupby(q, observed=True).apply(f).round(2).to_string(), sep='\n')
