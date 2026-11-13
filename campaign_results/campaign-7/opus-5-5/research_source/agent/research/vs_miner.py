"""Where does a given field miner beat us? (ours - miner)/p50 by quintile of realised/forecast vol,
of our forecast relative to the slow level, and by NY hour block."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
WS = Path(__file__).resolve().parents[2]; os.chdir(WS)
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.benchmark.generate_predictions import context_days_of, context_slice, load_minute_prices, load_module
ap = argparse.ArgumentParser()
ap.add_argument('--res'); ap.add_argument('--miner', type=int); ap.add_argument('--tl', type=int, default=3600)
ap.add_argument('--params', default='{}')
a = ap.parse_args()
mod = load_module(Path('agent/modeling.py')); mod.PARAMS.update(json.loads(a.params))
cd = context_days_of(mod); ti = 60 if a.tl == 3600 else 300; sm = ti // 60
d = pd.read_parquet(f'agent/research/results/{a.res}.parquet')
rows = []
for asset, g in d.groupby('asset'):
    st = pd.to_datetime(g.start_time, utc=True)
    sc = get_miner_scores(st.min().to_pydatetime(), (st.max() + pd.Timedelta(seconds=a.tl + 60)).to_pydatetime(), asset, a.tl, ti)
    m = sc[sc.miner_uid == a.miner].set_index('scored_time')['crps']
    prices = load_minute_prices(WS / 'market_data', asset, st.min() - pd.Timedelta(days=cd), st.max() + pd.Timedelta(seconds=a.tl))
    lpx = np.log(prices['close'])
    for (_, row), t in zip(g.iterrows(), st):
        if row.scored_time not in m.index:
            continue
        ctx = context_slice(prices, t, a.tl, cd)
        last, var, mow, dis = mod._forecast(asset, t.isoformat(), ti, a.tl, ctx)
        real = lpx.loc[t: t + pd.Timedelta(seconds=a.tl): sm].to_numpy()
        rv = np.sqrt(np.nanmean(np.diff(real) ** 2))
        rows.append({'asset': asset, 'd': (row.crps - m.loc[row.scored_time]) / row.p50, 'ratio': rv / np.sqrt(np.mean(var)),
                     'dis': dis, 'h': t.tz_convert('America/New_York').hour})
x = pd.DataFrame(rows)
print('mean (ours-miner)/p50 %:', round(100 * x.d.mean(), 3), 'n', len(x))
for col in ['ratio', 'dis']:
    q = pd.qcut(x[col], 5)
    print(col, (100 * x.groupby(q, observed=True).d.mean()).round(3).to_dict())
print('hour block', (100 * x.groupby(x.h // 4 * 4).d.mean()).round(3).to_dict())
