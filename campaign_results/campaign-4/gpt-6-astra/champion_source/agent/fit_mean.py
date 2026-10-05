"""A deliberately regularized drift test; training ends before validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality.json').read_text());models={};metrics=[]
for step in [60,300]:
 for asset in season['tables']:
  if step==60 and asset not in ['BTC','ETH','SOL','XRP','HYPE']:continue
  df=load_lenient(asset,pd.Timestamp('2026-06-01',tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
  log=np.log(df.close.resample(f'{step}s').last());r=log.diff()
  scale=r.abs().rolling(86400//step,min_periods=300//step).mean()*np.sqrt(np.pi/2)
  horizon=3600//step if step==60 else 86400//step
  feats={str(h):(log-log.shift(int(h*3600/step)))/(scale*np.sqrt(h*3600/step)) for h in [.25,1,6,24,72]}
  x=pd.DataFrame(feats).clip(-4,4)
  y=(log.shift(-horizon)-log)/(scale*np.sqrt(horizon))
  rows=x.copy();rows['y']=y;rows=rows.iloc[72*3600//step:-horizon:3600//step].dropna()
  X=rows.iloc[:,:-1].to_numpy();Y=rows.y.clip(-5,5).to_numpy()
  # No intercept: drift must be supported by signed recent-return features.
  beta=np.linalg.solve(X.T@X+np.eye(X.shape[1])*1000,X.T@Y)
  pred=X@beta
  models[f'{asset}_{step}']=beta.tolist()
  metrics.append(dict(asset=asset,step=step,n=len(rows),fit_r2=1-float(np.mean((Y-pred)**2)/np.mean(Y**2))))
Path('agent/mean.json').write_text(json.dumps(dict(fit_end='2026-07-31',models=models),indent=2))
print(pd.DataFrame(metrics).to_string(index=False))
