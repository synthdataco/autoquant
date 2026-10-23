"""Event effect: realised variance in windows after a scheduled release vs the same clock window on the
previous 10 weekdays without that event type (L2 ratio, and L1^2 ratio)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/workspace/agent/analysis')
from season import load
ev = pd.read_parquet('/workspace/market_data/calendar/events.parquet'); ev['t'] = pd.to_datetime(ev.time_utc, utc=True)
ev = ev[ev.t < '2026-10-11']
WINS = [(-30, 0), (0, 2), (2, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 240)]
MACRO = ('CPI','NFP','FOMC','PPI','GDP','PCE','RETAIL_SALES')
macro_days = set(ev[ev.event.isin(MACRO)].t.dt.normalize())
assets = sys.argv[1:]
out = []
for a in assets:
    c = load(a); r = np.log(c.ffill()).diff().fillna(0.0)
    days = set(r.index.normalize().unique())
    for _, e in ev.iterrows():
        if e.scope != 'all' and a not in e.scope.split(','): continue
        if e.event in ('NYSE_CLOSED', 'NYSE_EARLY_CLOSE', 'EARNINGS'): continue
        T = e.t
        if T - pd.Timedelta(days=15) < r.index[0]: continue
        same = set(ev[ev.event == e.event].t.dt.normalize())
        ctrl = [T - pd.Timedelta(days=k) for k in range(1, 20) if (T - pd.Timedelta(days=k)).dayofweek < 5 and (T - pd.Timedelta(days=k)).normalize() not in macro_days][:10]
        for w0, w1 in WINS:
            x = r[T + pd.Timedelta(minutes=w0): T + pd.Timedelta(minutes=w1 - 1)]
            cs = [r[t + pd.Timedelta(minutes=w0): t + pd.Timedelta(minutes=w1 - 1)] for t in ctrl]
            l2 = (x**2).sum() / np.mean([(q**2).sum() for q in cs])
            l1 = (x.abs().sum() / np.mean([q.abs().sum() for q in cs]))**2
            out.append((a, e.event, T, f'{w0}-{w1}', l2, l1))
d = pd.DataFrame(out, columns=['asset', 'event', 't', 'win', 'l2', 'l1'])
d.to_parquet('/tmp/events.parquet')
print(d.groupby(['event', 'win']).l1.agg(['median', 'mean', 'count']).unstack(1).round(2).to_string())
print(d[d.win == '0-2'].groupby(['asset', 'event']).l1.median().unstack().round(1).to_string())
