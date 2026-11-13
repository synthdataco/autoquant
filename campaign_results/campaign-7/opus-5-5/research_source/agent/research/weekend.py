"""Weekend activity over time: 5-min rms (bps) and zero share by NY time blocks, per weekend."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all
a = sys.argv[1]
df = load_all(a, start=sys.argv[2] if len(sys.argv) > 2 else '2026-05-01')
c = df['close']
r = (np.log(c).diff(5) * 1e4)
r = r[r.index.minute % 5 == 0]
ny = r.index.tz_convert('America/New_York')
blk = pd.Series(np.select([ (ny.day_of_week == 4) & (ny.hour >= 17), ny.day_of_week == 5, (ny.day_of_week == 6) & (ny.hour < 18), (ny.day_of_week == 6) & (ny.hour >= 18)],
                          ['Fri17+', 'Sat', 'Sun<18', 'Sun18+'], 'wk'), index=r.index)
wkd = (ny.day_of_week < 4) | ((ny.day_of_week == 4) & (ny.hour < 17))
week = (r.index - pd.to_timedelta(ny.day_of_week, unit='D')).floor('D')
d = pd.DataFrame({'r': r.values, 'blk': blk.values, 'week': week}).dropna()
g = d.groupby(['week', 'blk']).r.agg(lambda x: np.sqrt(np.mean(x**2))).unstack()
z = d.groupby(['week', 'blk']).r.agg(lambda x: np.mean(x == 0)).unstack()
out = g.round(2).astype(str) + ' (' + (100 * z).round(0).fillna(-1).astype(int).astype(str) + '%)'
print(a, '5-min rms bps (zero share) per week block')
print(out[['wk', 'Fri17+', 'Sat', 'Sun<18', 'Sun18+']].to_string())
