"""Event volatility diagnostics (not a substitute for official CRPS evaluation)."""
import os,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from research_model import shape_at
ap=argparse.ArgumentParser();ap.add_argument('--shrink',type=float,default=.1);ap.add_argument('--assets',nargs='+',default=['BTC','ETH','SOL','XRP','HYPE','XAU','SP500','WTIOIL','NVDAX','TSLAX','AAPLX','GOOGLX','SPCX']);ap.add_argument('--start',default='2026-01-01');ap.add_argument('--fine',action='store_true');ap.add_argument('--end',default='2026-09-01');ap.add_argument('--table',default='seasonality.json');ap.add_argument('--suffix',default=None);args=ap.parse_args()
suffix=args.suffix if args.suffix is not None else ('_fine' if args.fine else '')
cal=pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
cal=cal[(cal.time_utc<pd.Timestamp(args.end,tz='UTC')) & (cal.time_utc>=pd.Timestamp(args.start,tz='UTC'))]
rows=[]
for asset in args.assets:
 fs=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if args.start<=f.stem[5:]<args.end]
 df=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in fs]).set_index('timestamp').sort_index()
 r=np.log(df.close.astype(float)).diff();var=r*r/shape_at(r.index,asset,True,args.table,args.shrink)
 for e in cal.itertuples():
  if e.scope!='all' and asset not in e.scope.split(','):continue
  t=e.time_utc
  past=var.loc[t-pd.Timedelta(days=7):t-pd.Timedelta(minutes=1)]
  if past.notna().sum()<1000:continue
  level=past.mean()
  for lo,hi in ([(0,1),(1,5),(5,30),(30,120),(120,360)] if args.fine else [(0,5),(5,30),(30,120),(120,360)]):
   x=var.loc[t+pd.Timedelta(minutes=lo):t+pd.Timedelta(minutes=hi-1)]
   if x.notna().sum()<max(1,len(x)*.4):continue
   rows.append(dict(asset=asset,event=e.event,time=str(t),lo=lo,hi=hi,ratio=x.mean()/level))
res=pd.DataFrame(rows);res.to_csv(f'agent/results/event_diagnostics{suffix}.csv',index=False)
print(res[res.asset.isin(['BTC','ETH','SOL','XRP','HYPE'])].groupby(['event','lo']).ratio.agg(['count','median','mean']).round(2).to_string())
print('\nCOM/EQU\n',res[~res.asset.isin(['BTC','ETH','SOL','XRP','HYPE'])].groupby(['event','lo']).ratio.agg(['count','median','mean']).round(2).to_string())
