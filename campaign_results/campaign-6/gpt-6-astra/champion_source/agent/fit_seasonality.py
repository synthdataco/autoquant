"""Estimate recurring variance shape using only prices before the specified training end."""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from synth_lib.preparation.config import default_store_root
ASSETS=['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']
p=argparse.ArgumentParser();p.add_argument('--exclude-events');p.add_argument('--assets',nargs='+',default=ASSETS);p.add_argument('--bin-minutes',type=int,choices=[5,10,15,30],default=30);p.add_argument('--parent');p.add_argument('--parent-weight',type=float,default=.5);p.add_argument('--weekly',action='store_true');p.add_argument('--end',default='2026-09-01');p.add_argument('--start',default='2026-06-01');p.add_argument('--output',default='agent/seasonality.json');a=p.parse_args()
bins=1440//a.bin_minutes
parent=json.loads(Path(a.parent).read_text())['assets'] if a.parent else None
if parent and a.weekly:p.error('Fine-parent shrinkage and weekly refinement must be separate experiments')
out={'bin_minutes':a.bin_minutes,'parent':a.parent,'parent_weight':a.parent_weight if a.parent else 0.,'train_start':a.start,'train_end_exclusive':a.end,'assets':{}}
if a.exclude_events:
 event_fit=json.loads(Path(a.exclude_events).read_text())
 assert pd.Timestamp(event_fit['train_end_exclusive'])<=pd.Timestamp(a.end)
 calendar=pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
 calendar=calendar[(calendar.time_utc>=pd.Timestamp(a.start,tz='UTC')-pd.Timedelta(hours=3))&(calendar.time_utc<pd.Timestamp(a.end,tz='UTC'))]
 out.update(excluded_event_table=a.exclude_events,masked_return_fraction={})
for asset in a.assets:
 files=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if a.start<=f.stem[5:]<a.end]
 frame=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in files]).set_index('timestamp').sort_index()
 log=np.log(frame.close.astype(float));r=log.diff(5)
 if a.exclude_events:
  mask=np.zeros(len(r),dtype=bool)
  group='crypto' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'other'
  for event in calendar.itertuples():
   if event.scope!='all' and asset not in event.scope.split(','):continue
   for lo,mult in event_fit['groups'].get(group,{}).get(event.event,{}).items():
    if mult<=1:continue
    lo=int(lo);hi={0:5,5:30,30:120}[lo]
    start=event.time_utc+pd.Timedelta(minutes=lo);stop=event.time_utc+pd.Timedelta(minutes=hi)
    mask|=(r.index+pd.Timedelta(minutes=1)>start)&(r.index-pd.Timedelta(minutes=4)<stop)
  out['masked_return_fraction'][asset]=float(np.sum(mask&r.notna().to_numpy())/max(1,r.notna().sum()))
  r=r.mask(mask)
  print(asset,'masked scheduled-event return fraction',out['masked_return_fraction'][asset],flush=True)
 # Gaps are not filled. Both five-minute endpoints must be observed.
 daily=r.pow(2).resample('1D').mean();baseline=daily.rolling(14,min_periods=3,center=True).median()
 norm=baseline.reindex(r.index.floor('D')).to_numpy();z=r.to_numpy()/np.sqrt(norm)
 z=np.clip(z,-8,8)
 local=r.index.tz_convert('America/New_York');bucket=local.hour*(60//a.bin_minutes)+local.minute//a.bin_minutes;weekend=(local.dayofweek>=5).astype(int)
 df=pd.DataFrame({'z':z,'b':bucket,'w':weekend,'dow':local.dayofweek})
 groups=df.groupby(['w','b']).z
 var=groups.apply(lambda x: np.nanmean(x*x)).unstack().reindex(index=[0,1],columns=range(bins))
 absolute=groups.apply(lambda x:np.nanmean(np.abs(x))).unstack().reindex(index=[0,1],columns=range(bins))**2
 var=var/np.nanmean(var.to_numpy()[0]);absolute=absolute/np.nanmean(absolute.to_numpy()[0]);shape=.5*var+.5*absolute
 shape=shape.fillna(1).to_numpy();shape=.9*shape+.1*np.mean(shape[0]);shape/=np.mean(shape[0])
 if parent is not None:
  prior=np.repeat(np.asarray(parent[asset]),30//a.bin_minutes,axis=1)
  shape=(1-a.parent_weight)*shape+a.parent_weight*prior
  shape/=np.mean(shape[0])
 if a.weekly:
  weekly_var=df.groupby(['dow','b']).z.apply(lambda x:np.nanmean(x*x)).unstack().reindex(index=range(7),columns=range(bins))
  weekly_abs=df.groupby(['dow','b']).z.apply(lambda x:np.nanmean(np.abs(x))).unstack().reindex(index=range(7),columns=range(bins))**2
  weekly_var/=np.nanmean(weekly_var.to_numpy()[:5]);weekly_abs/=np.nanmean(weekly_abs.to_numpy()[:5])
  w=(.5*weekly_var+.5*weekly_abs).fillna(1).to_numpy()
  base=shape[(np.arange(7)>=5).astype(int)]
  w=.5*w+.5*base
  w/=w[:5].mean()
  out['assets'][asset]=w.tolist()
 else:out['assets'][asset]=shape.tolist()
 print(asset,'weekend/weekday vol',round(float(np.sqrt(np.mean(shape[1])/np.mean(shape[0]))),3),'intraday max/min vol',round(float(np.sqrt(shape[0].max()/shape[0].min())),2),'density',round(float(frame.close.notna().mean()),3),flush=True)
Path(a.output).write_text(json.dumps(out,indent=2))
