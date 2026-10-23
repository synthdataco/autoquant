"""Paired per-prompt comparison of two result CSVs, split by whether the horizon holds a macro release."""
import sys, pandas as pd, numpy as np
a, b = sys.argv[1], sys.argv[2]; hz = pd.Timedelta(seconds=int(sys.argv[3]) if len(sys.argv) > 3 else 86400)
A = pd.read_csv(a); B = pd.read_csv(b)
m = A.merge(B, on=['asset', 'scored_time'], suffixes=('_a', '_b'))
m['st'] = pd.to_datetime(m.scored_time, utc=True) - hz
ev = pd.read_parquet('/workspace/market_data/calendar/events.parquet'); ev['t'] = pd.to_datetime(ev.time_utc, utc=True)
ev = ev[ev.event.isin(['CPI', 'NFP', 'FOMC', 'PPI', 'GDP', 'PCE'])]
m['event'] = 'none'
for _, e in ev.iterrows():
    m.loc[(m.st <= e.t) & (m.st + hz > e.t), 'event'] = e.event
m['d'] = m.ours_b / m.ours_a - 1
print(m.groupby('event').agg(n=('d', 'size'), rel=('d', 'mean'), win=('d', lambda x: (x < 0).mean())).round(4).to_string())
print(m.groupby('asset').agg(n=('d', 'size'), rel=('d', 'mean'), meanA=('ours_a', 'mean'), meanB=('ours_b', 'mean')).round(4).to_string())
print('overall mean-CRPS change', round(m.ours_b.mean() / m.ours_a.mean() - 1, 5), ' per-asset mean of rel change', round((m.groupby('asset').ours_b.mean() / m.groupby('asset').ours_a.mean() - 1).mean(), 5))
