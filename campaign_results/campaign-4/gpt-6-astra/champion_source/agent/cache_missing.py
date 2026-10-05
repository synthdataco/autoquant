"""Recover allowed score/reward pages independently; enforce dates immediately."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time,requests
import pandas as pd
from synth.validator.competition_config import ALL_COMPETITIONS
from synth_lib.backtester.config import slug_for
root=Path('offline_scores');pages=root/'pages';pages.mkdir(exist_ok=True)
jobs=[]
start=pd.Timestamp('2026-09-02',tz='UTC');end=pd.Timestamp('2026-09-13',tz='UTC')
for c in ALL_COMPETITIONS:
 slug=slug_for(c)
 for a in c.asset_list:
  if (root/f'miner_scores_{a}_{slug}.parquet').exists():continue
  for day in pd.date_range(start,end,freq='D',inclusive='left'):jobs.append(('scores',slug,a,c.time_length,c.time_increment,day))
 if not (root/f'rewards_history_{slug}.parquet').exists():
  for day in pd.date_range(start,end,freq='D',inclusive='left'):jobs.append(('rewards',slug,None,c.time_length,c.time_increment,day))
def fetch(job):
 kind,slug,asset,horizon,step,day=job;p=pages/f'{kind}_{slug}_{asset}_{day.date()}.parquet'
 if p.exists():return str(p),'exists'
 nxt=day+pd.Timedelta(days=1)
 params={'from':day.strftime('%Y-%m-%dT%H:%M:%SZ'),'to':nxt.strftime('%Y-%m-%dT%H:%M:%SZ')}
 if kind=='scores':params.update(asset=asset,time_length=horizon,time_increment=step);endpoint='validation/scores/historical';column='scored_time'
 else:params['prompt_name']=slug;endpoint='rewards/scores';column='updated_at'
 try:
  r=requests.get('https://api.synthdata.co/'+endpoint,params=params,timeout=8);r.raise_for_status();d=pd.DataFrame(r.json())
  if not d.empty:
   d[column]=pd.to_datetime(d[column],utc=True);d=d.loc[(d[column]>=day)&(d[column]<nxt)].copy()
  d.to_parquet(p,index=False);return str(p),len(d)
 except requests.RequestException as exc:return str(p),type(exc).__name__
with ThreadPoolExecutor(max_workers=2) as pool:
 for f in as_completed([pool.submit(fetch,j) for j in jobs]):print(f.result(),flush=True)
for c in ALL_COMPETITIONS:
 slug=slug_for(c)
 for asset in c.asset_list:
  target=root/f'miner_scores_{asset}_{slug}.parquet'
  fs=list(pages.glob(f'scores_{slug}_{asset}_*.parquet'))
  if not target.exists() and len(fs)==11:
   d=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True).drop_duplicates();d['time_increment']=c.time_increment;d['start_time']=d.scored_time-pd.Timedelta(seconds=c.time_length);d.to_parquet(target,index=False)
 target=root/f'rewards_history_{slug}.parquet';fs=list(pages.glob(f'rewards_{slug}_None_*.parquet'))
 if not target.exists() and len(fs)==11:pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True).drop_duplicates().to_parquet(target,index=False)
