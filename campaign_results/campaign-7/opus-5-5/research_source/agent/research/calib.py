"""Calibration diagnostic (not CRPS): standardised 5-min (or 1-min) step returns u = r/sigma_model,
mean(u^2) by NY weekday x hour-block, per asset; prompts every --every minutes."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
ap = argparse.ArgumentParser()
ap.add_argument('--assets', required=True); ap.add_argument('--tl', type=int, default=86400)
ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
ap.add_argument('--every', type=int, default=180); ap.add_argument('--params', default=None)
ap.add_argument('--by', default='dow')
a = ap.parse_args()
mod = load_module(Path('agent/modeling.py'))
if a.params: mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod); ti = 300 if a.tl == 86400 else 60; sm = ti // 60
rows = []
for asset in a.assets.split(','):
    s, e = pd.Timestamp(a.start, tz='UTC'), pd.Timestamp(a.end, tz='UTC')
    prices = load_minute_prices(WS / 'market_data', asset, s - pd.Timedelta(days=cd), e + pd.Timedelta(seconds=a.tl))
    lpx = np.log(prices['close'])
    for t in pd.date_range(s, e, freq=f'{a.every}min', inclusive='left'):
        ctx = context_slice(prices, t, a.tl, cd)
        last, var, mow, _ = mod._forecast(asset, t.isoformat(), ti, a.tl, ctx)
        real = lpx.loc[t: t + pd.Timedelta(seconds=a.tl): sm].to_numpy()
        r = np.diff(real)
        u2 = r ** 2 / var
        st = pd.date_range(t, periods=len(r), freq=f'{sm}min').tz_convert('America/New_York')
        rows.append(pd.DataFrame({'asset': asset, 'u2': u2, 'dow': st.day_of_week, 'h': st.hour, 'lead': np.arange(len(r)) * sm}))
d = pd.concat(rows).dropna()
d['hb'] = d.h // 4 * 4
print('overall mean u2:', d.groupby('asset').u2.mean().round(2).to_dict())
if a.by == 'dow':
    print((d.pivot_table(index='asset', columns='dow', values='u2', aggfunc='mean')).round(2).to_string())
    print((d.pivot_table(index='asset', columns='hb', values='u2', aggfunc='mean')).round(2).to_string())
else:
    d['lb'] = pd.cut(d.lead, [-1, 30, 120, 360, 720, 1440] if a.tl == 86400 else [-1, 5, 15, 30, 60])
    print((d.pivot_table(index='asset', columns='lb', values='u2', aggfunc='mean', observed=True)).round(2).to_string())
