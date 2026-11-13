"""Event effect sizes (exploration): realised variance in the m minutes after each event, over the
model-implied variance (deseasonalised level from the previous 24h x v1 profile). Per type/asset:
median and mean of the ratio, over occurrences before the cutoff."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS
prof = np.load('agent/research/profiles_full.npz')
ev = pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
ev['t'] = pd.to_datetime(ev.time_utc, utc=True)
ev = ev[ev.t < '2026-11-08']
M = [1, 5, 30, 120]
rows = []
for a in ALL_ASSETS:
    df = load_all(a)
    lp = np.log(df['close'])
    r1 = lp.diff().shift(-1)  # return of bar t: close(t+1)/close(t)... bar labelled t covers [t, t+1)
    r1 = (lp.shift(-1) - lp)  # r at label t = move from close(t) to close(t+1)
    ny = df.index.tz_convert('America/New_York')
    mow = (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()
    p1 = np.maximum(prof[f'{a}_v1'][mow], 0.03)
    z = (r1.to_numpy() ** 2) / p1
    zs = pd.Series(z, index=df.index)
    for _, e in ev.iterrows():
        if not (e.scope == 'all' or a in e.scope.split(',')):
            continue
        t = e.t
        # the event's first price reaction is the close of bar t (covers [t, t+1)); return label t-1 -> close(t-1)->close(t)
        t0 = t - pd.Timedelta(minutes=1)
        if t0 - pd.Timedelta(days=1) < df.index[0] or t0 + pd.Timedelta(minutes=max(M)) > df.index[-1]:
            continue
        base = zs.loc[t0 - pd.Timedelta(days=1): t0 - pd.Timedelta(minutes=1)]
        lvl = np.nanmean(np.minimum(base, 30 * np.nanmedian(base)))
        if not np.isfinite(lvl) or lvl <= 0:
            continue
        rec = {'asset': a, 'event': e.event, 't': t}
        for m in M:
            seg = slice(t0, t0 + pd.Timedelta(minutes=m - 1))
            real = np.nansum(r1.loc[seg].to_numpy() ** 2)
            expv = lvl * np.sum(p1[df.index.get_indexer(r1.loc[seg].index)])
            rec[f'R{m}'] = real / expv
        rows.append(rec)
d = pd.DataFrame(rows)
d.to_parquet('agent/research/results/event_ratios.parquet')
grp = 'crypto' 
d['grp'] = np.where(d.asset.isin(['BTC', 'ETH', 'SOL', 'XRP', 'HYPE']), 'crypto', 'equ')
pd.set_option('display.width', 200)
print(d.groupby(['event', 'grp'])[[f'R{m}' for m in M]].median().round(2).to_string())
print('MEAN')
print(d.groupby(['event', 'grp'])[[f'R{m}' for m in M]].mean().round(2).to_string())
print(d.groupby(['event', 'grp']).size().to_string())
