import requests
from pathlib import Path
import pandas as pd
root=Path('offline_scores');pages=root/'pages'
for asset,day in [('NVDAX','2026-09-04'),('WTIOIL','2026-09-11')]:
 st=pd.Timestamp(day,tz='UTC');en=st+pd.Timedelta(days=1)
 try:
  r=requests.get('https://api.synthdata.co/validation/scores/historical',params={'from':st.strftime('%Y-%m-%dT%H:%M:%SZ'),'to':en.strftime('%Y-%m-%dT%H:%M:%SZ'),'asset':asset,'time_length':86400,'time_increment':300},timeout=15);r.raise_for_status();df=pd.DataFrame(r.json());df['scored_time']=pd.to_datetime(df.scored_time,utc=True);df=df.loc[(df.scored_time>=st)&(df.scored_time<en)];df.to_parquet(pages/f'scores_com-equ-24h_{asset}_{day}.parquet',index=False)
  fs=[pages/f'scores_com-equ-24h_{asset}_{d.date()}.parquet' for d in pd.date_range('2026-09-04','2026-09-12')]
  df=pd.concat([pd.read_parquet(p) for p in fs],ignore_index=True).drop_duplicates();df['time_increment']=300;df['start_time']=df.scored_time-pd.Timedelta(days=1);df.to_parquet(root/f'miner_scores_{asset}_com-equ-24h.parquet',index=False);print(asset,len(df),flush=True)
 except requests.RequestException as exc:print(asset,type(exc).__name__,flush=True)
