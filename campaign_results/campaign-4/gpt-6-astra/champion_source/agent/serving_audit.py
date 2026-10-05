import time,json
from pathlib import Path
import numpy as np,pandas as pd
from modeling import simulate,_checkpoint
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd();rows=[]
for asset in 'BTC ETH SOL XRP HYPE XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500'.split():
 ctx=load_lenient(asset,pd.Timestamp('2026-09-08',tz='UTC'),pd.Timestamp('2026-09-14 23:59',tz='UTC'))
 for horizon,step in [(86400,300)]+([(3600,60)] if asset in ['BTC','ETH','SOL','XRP','HYPE'] else []):
  start=pd.Timestamp('2026-09-15 00:05',tz='UTC');tick=time.perf_counter();out=simulate(asset,start.isoformat(),step,horizon,1000,ctx);elapsed=time.perf_counter()-tick
  a=np.asarray(out[2:]);assert a.shape==(1000,horizon//step+1) and np.isfinite(a).all() and (a>0).all()
  assert (a[:,0]==ctx.close.dropna().iloc[-1]).all()
  rows.append(dict(asset=asset,horizon=horizon,latency_seconds=elapsed,fit_end=_checkpoint(start,horizon)['fit_end'],paths=a.shape[0],points=a.shape[1]))
pd.DataFrame(rows).to_csv('agent/results/serving_audit.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
