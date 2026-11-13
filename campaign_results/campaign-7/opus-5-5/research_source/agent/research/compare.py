"""Paired comparison of result files: per-asset % change of mean CRPS vs a reference tag."""
import sys
import pandas as pd, numpy as np
R = 'agent/research/results/'
comp = sys.argv[1]; ref = sys.argv[2]; others = sys.argv[3:]
base = pd.read_parquet(f'{R}{ref}_{comp}.parquet').set_index(['asset', 'scored_time'])['crps']
rows = {}
for o in others:
    d = pd.read_parquet(f'{R}{o}_{comp}.parquet').set_index(['asset', 'scored_time'])['crps']
    j = pd.concat([base.rename('a'), d.rename('b')], axis=1, join='inner')
    per = j.groupby(level=0).apply(lambda x: 100 * (x.b.mean() / x.a.mean() - 1))
    # day-clustered t-stat of the relative difference
    j['day'] = j.index.get_level_values(1).floor('D')
    dd = j.groupby('day').apply(lambda x: x.b.sum() / x.a.sum() - 1)
    tstat = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 2 else np.nan
    per['ALL(mean%)'] = per.mean()
    per['better_cells'] = int((per.drop('ALL(mean%)') < 0).sum())
    per['t_day'] = tstat
    rows[o] = per
print(f'{comp}: % change in mean CRPS vs {ref} (negative = better)')
print(pd.DataFrame(rows).T.round(2).to_string())
