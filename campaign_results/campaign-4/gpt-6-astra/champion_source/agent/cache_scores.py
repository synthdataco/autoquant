"""Fetch allowed historical field scores and rewards ONLY, never market prices.
All requests end on/before the frozen cutoff. This repairs the missing offline bundle.
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
from synth_lib.backtester.loading import get_miner_scores,get_rewards_history
import synth_lib.backtester.loading as loading
loading.API_SCORES_PAGE_SIZE_DAYS=1
import requests
def bounded_get(url,params,timeout=12):
    return requests.get(url,params=params,timeout=12)
loading._http_get=bounded_get
from synth_lib.backtester.config import slug_for
from synth.validator.competition_config import ALL_COMPETITIONS
assert not os.environ.get('SYNTH_BACKTESTER_OFFLINE_DATA_ROOT')
root=Path('offline_scores');root.mkdir(exist_ok=True)
start=pd.Timestamp('2026-09-02',tz='UTC').to_pydatetime()
end=pd.Timestamp('2026-09-13',tz='UTC').to_pydatetime()
jobs=[(c,a) for c in ALL_COMPETITIONS for a in c.asset_list]
def fetch(job):
 c,a=job;p=root/f'miner_scores_{a}_{slug_for(c)}.parquet'
 if p.exists():return p.name,'exists'
 try:
  df=get_miner_scores(start,end,a,c.time_length,c.time_increment)
 except Exception as exc:
  return p.name,type(exc).__name__
 # Some historical endpoints over-return their requested boundary. Discard
 # unsolicited rows before using any scores; do not inspect their values.
 if not df.empty:
  df=df.loc[(df.scored_time>=pd.Timestamp(start))&(df.scored_time<=pd.Timestamp(end))].copy()
 df.to_parquet(p,index=False)
 return p.name,len(df)
with ThreadPoolExecutor(max_workers=2) as pool:
 for f in as_completed([pool.submit(fetch,j) for j in jobs]):print(f.result(),flush=True)
for c in ALL_COMPETITIONS:
 p=root/f'rewards_history_{slug_for(c)}.parquet'
 if p.exists():continue
 df=get_rewards_history(start,end,slug_for(c))
 if not df.empty:
  times=pd.to_datetime(df.updated_at,utc=True)
  df=df.loc[(times>=pd.Timestamp(start))&(times<=pd.Timestamp(end))].copy()
 df.to_parquet(p,index=False)
 print(p.name,len(df),flush=True)
