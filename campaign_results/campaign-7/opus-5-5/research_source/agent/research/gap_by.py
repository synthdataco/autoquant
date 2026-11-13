"""Where is the gap to the field? ratio sum(crps)/sum(p10) by NY hour and by day."""
import sys
import pandas as pd, numpy as np
d = pd.read_parquet(f'agent/research/results/{sys.argv[1]}.parquet')
d['st'] = pd.to_datetime(d.start_time, utc=True)
ny = d.st.dt.tz_convert('America/New_York')
d['nyh'] = ny.dt.hour; d['dow'] = ny.dt.dayofweek; d['day'] = d.st.dt.floor('D')
f = lambda x: 100 * (x.crps.sum() / x.p10.sum() - 1)
h = d.groupby('nyh').apply(f)
print('by NY hour  :', ' '.join(f'{i}:{v:.1f}' for i, v in h.items()))
w = d.groupby('dow').apply(f)
print('by weekday  :', ' '.join(f'{i}:{v:.1f}' for i, v in w.items()))
dd = d.groupby('day').apply(f)
print('by day      :', ' '.join(f'{i.day}:{v:.1f}' for i, v in dd.items()))
# by own CRPS quintile (relative to the prompt's p10): heavy-tail prompts?
d['rel'] = d.crps / d.p10 - 1
q = pd.qcut(d.p10.rank(pct=True), 5, labels=False)
print('by p10 quintile (calm->wild):', d.groupby(q).apply(f).round(2).to_dict())
print('median rel%', round(100 * d.rel.median(), 2), 'mean rel%', round(100 * d.rel.mean(), 2))
