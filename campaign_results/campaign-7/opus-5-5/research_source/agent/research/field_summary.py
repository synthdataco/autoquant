"""Summarize the field's CRPS per competition/asset from the offline bundle."""
import glob, sys
import pandas as pd, numpy as np
root = 'market_data/offline_data/miner_scores'
rows = []
for comp in ['crypto-24h', 'com-equ-24h', 'crypto-1h']:
    for d in sorted(glob.glob(f'{root}/{comp}/*')):
        df = pd.read_parquet(d)
        df = df[df.scored_time >= '2026-08-03']
        df = df[df.crps > 0]
        g = df.groupby('scored_time')['crps']
        q = pd.DataFrame({'best': g.min(), 'p10': g.quantile(0.1), 'p50': g.median(), 'n': g.size()})
        st = df.drop_duplicates('scored_time').set_index('scored_time')['start_time']
        mins = st.dt.minute.value_counts().head(3).to_dict()
        rows.append(dict(comp=comp, asset=d.split('/')[-1], prompts=len(q), miners=int(q.n.median()),
                         best=q.best.mean(), p10=q.p10.mean(), p50=q.p50.mean(),
                         p10_over_best=(q.p10/q.best).mean()-1, p50_over_best=(q.p50/q.best).mean()-1,
                         start_minutes=mins))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(pd.DataFrame(rows).round(3).to_string())
