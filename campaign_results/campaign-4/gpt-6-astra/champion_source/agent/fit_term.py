import json
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality_abs48.json').read_text());models={};metrics=[]
for asset in season['tables']:
 begin='2026-01-01' if asset in ['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX'] else '2026-06-01'
 df=load_lenient(asset,pd.Timestamp(begin,tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
 for step in [60,300] if asset in ['BTC','ETH','SOL','XRP','HYPE'] else [300]:
  close=df.close.resample(f'{step}s').last();r=np.log(close).diff()
  shape=np.array(season['tables'][asset]);norm=np.sqrt(shape[close.index.dayofweek*48+(close.index.hour*60+close.index.minute)//30]);a=r.abs()/norm
  features=pd.DataFrame({str(h):a.rolling(max(1,int(h*3600/step)),min_periods=max(1,int(h*3600/step)//3)).mean() for h in [.25,1,6,24,72,168]})
  edges=[0,5,15,30,60] if step==60 else [0,60,180,360,720,1440]
  fits=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   offset=lo*60//step;length=(hi-lo)*60//step
   target=a.shift(-offset-1).iloc[::-1].rolling(length,min_periods=max(1,int(length*.75))).mean().iloc[::-1]
   rows=features.copy();rows['target']=target;rows=rows.iloc[168*3600//step:-edges[-1]*60//step:3600//step].dropna()
   vals=rows.to_numpy();vals=vals[(vals>1e-10).all(axis=1)]
   X=np.log(vals[:,:-1]);y=np.log(vals[:,-1]);center=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1)
   xx=np.c_[np.ones(len(X)),(X-center)/sd];reg=np.eye(7)*50;reg[0,0]=0
   beta=np.linalg.solve(xx.T@xx+reg,xx.T@y);res=y-xx@beta
   fits.append(dict(center=center.tolist(),sd=sd.tolist(),beta=beta.tolist(),correction=float(np.exp(res).mean())))
  models[f'{asset}_{step}']=dict(edges=edges,fits=fits)
  print(asset,step,len(X),flush=True)
Path('agent/term.json').write_text(json.dumps(dict(fit_end='2026-07-31',models=models),indent=2))
