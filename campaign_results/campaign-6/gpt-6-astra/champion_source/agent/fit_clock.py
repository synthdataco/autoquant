"""Diagnose within-hour volatility seasonality before validation."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from research_model import shape_at
p=argparse.ArgumentParser();p.add_argument('--start',default='2026-06-01');p.add_argument('--end',default='2026-09-01');p.add_argument('--table',default='seasonality.json');p.add_argument('--output',default='agent/clock.json');a=p.parse_args()
out={}
for asset in ['BTC','ETH','SOL','XRP','HYPE']:
 fs=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if a.start<=f.stem[5:]<a.end]
 df=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in fs]).set_index('timestamp').sort_index()
 r=np.log(df.close.astype(float)).diff();r2=r*r
 norm=r2.rolling(1440*7,min_periods=1440).mean();z2=(r2/norm).clip(upper=64)/shape_at(r.index,asset,True,a.table)
 bymin=z2.groupby(z2.index.minute).mean();bymin/=bymin.mean()
 byabs=np.sqrt(z2).groupby(z2.index.minute).mean()**2;byabs/=byabs.mean()
 shape=.5*bymin+.5*byabs;shape=.75*shape+.25
 out[asset]=shape.to_list()
 print(asset,'first minute variance',round(shape.iloc[0],2),'minute30',round(shape.iloc[30],2),'first5/rest ratio',round(shape.iloc[:5].mean()/shape.iloc[5:].mean(),2),flush=True)
Path(a.output).write_text(json.dumps({'train_start':a.start,'train_end_exclusive':a.end,'assets':out},indent=2))
