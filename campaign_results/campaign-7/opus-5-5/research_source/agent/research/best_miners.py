"""How do individual field miners compare with the per-prompt p10? (sum crps / sum p10 - 1)"""
import sys
import pandas as pd, numpy as np
comp, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
assets = {'crypto-1h': ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'], 'crypto-24h': ['BTC', 'ETH', 'SOL', 'XRP', 'HYPE'],
          'com-equ-24h': ['XAU', 'NVDAX', 'GOOGLX', 'TSLAX', 'AAPLX', 'WTIOIL', 'SPCX', 'SP500']}[comp]
out = []
for a in assets:
    df = pd.read_parquet(f'market_data/offline_data/miner_scores/{comp}/{a}')
    df = df[(df.scored_time >= start) & (df.scored_time < end) & (df.crps > 0)]
    p10 = df.groupby('scored_time').crps.quantile(0.1).rename('p10')
    df = df.join(p10, on='scored_time')
    n = df.scored_time.nunique()
    g = df.groupby('miner_uid').agg(cnt=('crps', 'size'), c=('crps', 'sum'), p=('p10', 'sum'))
    g = g[g.cnt >= 0.98 * n]
    g['rel'] = 100 * (g.c / g.p - 1)
    g['asset'] = a
    out.append(g.reset_index())
x = pd.concat(out)
piv = x.pivot_table(index='miner_uid', columns='asset', values='rel')
piv = piv.dropna()
piv['mean'] = piv.mean(axis=1)
print(f'{comp} {start}..{end}: miners with full coverage: {len(piv)}; best 8 by mean rel-to-p10 (%):')
print(piv.sort_values('mean').head(8).round(2).to_string())
print('quantiles of mean:', piv['mean'].quantile([0.0, 0.05, 0.1, 0.25, 0.5]).round(2).to_dict())
