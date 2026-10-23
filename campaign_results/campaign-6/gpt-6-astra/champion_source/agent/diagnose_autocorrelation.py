"""Prevalidation base-step return autocorrelation; no price interpolation or scores."""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from research_model import shape_at,clock_at
p=argparse.ArgumentParser();p.add_argument('--start',default='2026-06-01');p.add_argument('--end',default='2026-09-01');p.add_argument('--split',default='2026-08-01');p.add_argument('--coarse',default='seasonality.json');p.add_argument('--fine',default='seasonality_fine.json');p.add_argument('--clock',default='clock.json');p.add_argument('--output',default='agent/autocorrelation.json');p.add_argument('--diagnostics',default='agent/results/autocorrelation_diagnostics.csv');p.add_argument('--assets',nargs='+');a=p.parse_args()
warm_start=(pd.Timestamp(a.start)+pd.Timedelta(days=7)).strftime('%Y-%m-%d')
crypto=['BTC','ETH','SOL','XRP','HYPE'];assets=crypto+['XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']
if a.assets:assets=a.assets
rows=[];out={}
for asset in assets:
 fs=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if a.start<=f.stem[5:]<a.end]
 frame=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in fs]).set_index('timestamp').sort_index()
 out[asset]={}
 for lag in ([1,5] if asset in crypto else [5]):
  shape=shape_at(frame.index,asset,True,a.coarse if asset in crypto else a.fine)
  if asset in crypto:shape*=clock_at(frame.index,asset,lag,a.clock)
  r=np.log(frame.close.astype(float)).diff(lag)
  rv=r.pow(2)/lag/shape
  level=rv.ewm(halflife=4320,min_periods=1440).mean().shift(lag)
  z=(r/np.sqrt(lag*shape*level)).clip(-8,8)
  pairs=pd.DataFrame({'current':z,'previous':z.shift(lag)}).loc[warm_start:].dropna()
  def correlation(d):return float(d.current.corr(d.previous))
  early=correlation(pairs[pairs.index<pd.Timestamp(a.split,tz='UTC')]);late=correlation(pairs[pairs.index>=pd.Timestamp(a.split,tz='UTC')]);raw=correlation(pairs);rho=float(np.clip(.5*raw,-.1,.1))
  out[asset][str(lag)]=rho
  rows.append(dict(asset=asset,lag=lag,pairs=len(pairs),early_rho=early,august_rho=late,raw_rho=raw,shrunk_rho=rho,long_std_factor=float(np.sqrt((1+rho)/(1-rho)))))
  print(rows[-1],flush=True)
Path(a.output).write_text(json.dumps({'train_start':warm_start,'train_end_exclusive':a.end,'assets':out},indent=2))
pd.DataFrame(rows).to_csv(a.diagnostics,index=False)
