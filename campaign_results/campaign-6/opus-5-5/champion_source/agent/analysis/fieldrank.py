"""Where a results CSV sits among individual miners by mean CRPS over the same scored prompts (CRPS, not reward rank)."""
import sys, pandas as pd, numpy as np
f, comp = sys.argv[1], sys.argv[2]
d = pd.read_csv(f); d['scored_time'] = pd.to_datetime(d.scored_time, utc=True)
for asset, g in d.groupby('asset', sort=False):
    fs = pd.read_parquet(f'market_data/offline_data/miner_scores_{asset}_{comp}.parquet')
    fs['scored_time'] = pd.to_datetime(fs.scored_time, utc=True)
    fs = fs[fs.scored_time.isin(g.scored_time) & (fs.crps > 0)]
    cov = fs.groupby('miner_uid').scored_time.nunique()
    full = cov[cov >= 0.95 * g.scored_time.nunique()].index
    mm = fs[fs.miner_uid.isin(full)].groupby('miner_uid').crps.mean().sort_values()
    ours = g.ours.mean()
    rank = (mm < ours).sum() + 1
    print(f"{asset:7s} ours {ours:9.2f} | best miner {mm.iloc[0]:9.2f} ({ours/mm.iloc[0]-1:+.2%}) 5th {mm.iloc[4]:9.2f} 20th {mm.iloc[19]:9.2f} median {mm.median():9.2f} | rank {rank}/{len(mm)}")
