"""Separate model interactions from the use of longer validation windows.

All inputs are previously official-scored prompt CRPS; no metric recomputation.
"""
from pathlib import Path
import pandas as pd
rows=[]
for label,new,base in [('original','shrink02','seasonal'),('v5','floor02_v5_float32','candidate_v5_float32'),('v7','floor02_v7_float32','candidate_v7_float32')]:
 for f in (Path('agent/results')/new).glob('*.parquet'):
  asset=f.stem.split('_')[0]
  if asset in ['BTC','ETH','SOL','XRP','HYPE']:continue
  b=Path('agent/results')/base/f.name
  if not b.exists():continue
  keys=['asset','time_length','start_time','scored_time']
  x=pd.read_parquet(f);y=pd.read_parquet(b)
  d=x[keys+['crps']].merge(y[keys+['crps']],on=keys,suffixes=('_new','_base'),validate='one_to_one')
  if len(d)!=len(x) or len(d)!=len(y):raise ValueError('Prompt mismatch')
  time=pd.to_datetime(d.scored_time,utc=True)
  initial=((time>=pd.Timestamp('2026-09-14',tz='UTC'))&(time<pd.Timestamp('2026-09-16',tz='UTC')))|((time>=pd.Timestamp('2026-10-08',tz='UTC'))&(time<pd.Timestamp('2026-10-10',tz='UTC')))
  for part,mask in [('initial_dates',initial),('additional_dates',~initial)]:
   q=d[mask]
   if q.empty:continue
   rows.append(dict(model=label,asset=asset,month=time.iloc[0].strftime('%Y-%m'),part=part,n=len(q),change_pct=100*(q.crps_new.mean()/q.crps_base.mean()-1)))
d=pd.DataFrame(rows)
d.to_csv('agent/results/floor_window_attribution.csv',index=False)
print(d.groupby(['model','month','part']).agg(pct=('change_pct','mean'),wins=('change_pct',lambda x:int((x<0).sum())),cells=('change_pct','size')).to_string())
