import json
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
from ohlcv_features import features
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality_abs48.json').read_text());models={};metrics=[]
for asset in season['tables']:
    begin='2026-01-01' if asset in ['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX'] else '2026-06-01'
    df=load_lenient(asset,pd.Timestamp(begin,tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
    for step in [60,300] if asset in ['BTC','ETH','SOL','XRP','HYPE'] else [300]:
        x,a=features(df,step,season,asset)
        horizon=3600//step if step==60 else 86400//step
        target=a.shift(-1).iloc[::-1].rolling(horizon,min_periods=max(1,int(.75*horizon))).mean().iloc[::-1]
        rows=x.copy();rows['y']=np.log(target.clip(lower=1e-10));rows=rows.iloc[168*3600//step:-horizon:3600//step].replace([np.inf,-np.inf],np.nan).dropna()
        y=rows.pop('y').to_numpy();X=rows.to_numpy();center=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1)
        xx=np.c_[np.ones(len(X)),(X-center)/sd]
        reg=np.eye(xx.shape[1])*100;reg[0,0]=0
        beta=np.linalg.solve(xx.T@xx+reg,xx.T@y);res=y-xx@beta
        correction=float(np.exp(res).mean())
        models[f'{asset}_{step}']=dict(center=center.tolist(),sd=sd.tolist(),beta=beta.tolist(),correction=correction)
        metrics.append(dict(asset=asset,step=step,n=len(X),log_rmse=float(res.std())))
Path('agent/ohlcv.json').write_text(json.dumps(dict(fit_end='2026-07-31',models=models),indent=2))
print(pd.DataFrame(metrics).to_string(index=False))
