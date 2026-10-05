"""Measure whether announcement-hour variance differs from ordinary sessions.
Only pre-validation snapshot observations are used; no evaluation-window prices.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality_abs48.json').read_text())
# Past FOMC decision dates, fixed public calendar (not inferred from price moves).
dates=['2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29']
rows=[]
for asset in ['BTC','ETH','SOL','XRP','HYPE']:
 df=load_lenient(asset,pd.Timestamp('2026-01-01',tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
 r=np.log(df.close).diff()
 for day in dates:
  t=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=19 if day<'2026-03-08' else 18)
  event=r.loc[t:t+pd.Timedelta(minutes=59)].abs().mean()
  controls=[]
  for weeks in [-2,-1,1,2]:
   c=t+pd.Timedelta(days=7*weeks)
   x=r.loc[c:c+pd.Timedelta(minutes=59)].abs()
   if len(x)>50:controls.append(x.mean())
  rows.append(dict(asset=asset,date=day,event_abs=event,control_abs=np.mean(controls),ratio=event/np.mean(controls)))
d=pd.DataFrame(rows);d.to_csv('agent/results/fomc_exploration.csv',index=False)
print(d.to_string(index=False));print('Median ratio',d.ratio.median())
