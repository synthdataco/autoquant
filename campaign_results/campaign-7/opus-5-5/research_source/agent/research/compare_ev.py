"""Paired % change of CRPS restricted to prompts whose horizon contains an event of each type."""
import os, sys
import numpy as np, pandas as pd
R = 'agent/research/results/'
comp, ref, other = sys.argv[1], sys.argv[2], sys.argv[3]
H = pd.Timedelta(hours=1) if comp == 'crypto-1h' else pd.Timedelta(hours=24)
a = pd.read_parquet(f'{R}{ref}_{comp}.parquet').set_index(['asset', 'scored_time'])['crps']
b = pd.read_parquet(f'{R}{other}_{comp}.parquet').set_index(['asset', 'scored_time'])['crps']
j = pd.concat([a.rename('a'), b.rename('b')], axis=1, join='inner').reset_index()
j['st'] = j.scored_time - H
ev = pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR']); ev['t'] = pd.to_datetime(ev.time_utc, utc=True)
any_ev = np.zeros(len(j), bool)
for typ, g in ev.groupby('event'):
    m = np.zeros(len(j), bool)
    for _, e in g.iterrows():
        sc = (j.asset.isin(e.scope.split(','))) if e.scope != 'all' else np.ones(len(j), bool)
        m |= sc & (j.st < e.t) & (j.st + H > e.t)
    any_ev |= m
    if m.sum():
        print(f'{typ:15s} n={m.sum():5d}  change {100 * (j.b[m].sum() / j.a[m].sum() - 1):+.2f}%')
print(f'{"no event":15s} n={(~any_ev).sum():5d}  change {100 * (j.b[~any_ev].sum() / j.a[~any_ev].sum() - 1):+.2f}%')
