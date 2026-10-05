"""Fit a regularized heterogeneous autoregression of future absolute returns.
This is a training loss, not a replacement CRPS evaluation.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
season=json.loads(Path('agent',os.environ.get('SEASON_FILE','seasonality.json')).read_text())
assets=list(season['tables'])
artifacts={};metrics=[]
fit_end=os.environ.get('FIT_END','2026-07-31')
for step in [60,300]:
 for asset in assets:
    if step==60 and asset not in ['BTC','ETH','SOL','XRP','HYPE']:continue
    begin='2026-01-01' if asset in ['BTC','ETH','SOL','XRP','HYPE','NVDAX','GOOGLX','TSLAX','AAPLX','XAU'] else '2026-06-01'
    if asset in ['BTC','ETH','SOL','XRP','HYPE']:
        begin=os.environ.get('CRYPTO_TRAIN_START',begin)
    df=load_lenient(asset,pd.Timestamp(begin,tz='UTC'),pd.Timestamp(fit_end+' 23:59',tz='UTC'))
    close=df.close.resample(f'{step}s').last()
    r=np.log(close).diff()
    shape=np.asarray(season['tables'][asset])
    bins=season.get('bins',24)
    norm=np.sqrt(shape[close.index.dayofweek*bins+(close.index.hour*60+close.index.minute)//(1440//bins)])
    a=r.abs()/norm
    # Absolute-return averages provide a robust scale estimate.
    windows=[max(1,int(h*3600/step)) for h in [.25,1,6,24,72,168]]
    features=pd.DataFrame({str(h):a.rolling(w,min_periods=max(1,w//3)).mean() for h,w in zip([.25,1,6,24,72,168],windows)})
    target_window=3600//step if step==60 else 86400//step
    target=a.shift(-1).iloc[::-1].rolling(target_window,min_periods=int(target_window*.75)).mean().iloc[::-1]
    rows=features.copy();rows['target']=target
    rows=rows.iloc[windows[-1]:-target_window:3600//step].dropna()
    vals=rows.to_numpy(); good=(vals>1e-10).all(axis=1);vals=vals[good]
    X=np.log(vals[:,:-1]);y=np.log(vals[:,-1])
    center=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1)
    xx=np.column_stack([np.ones(len(X)),(X-center)/sd])
    reg=np.eye(xx.shape[1])*30;reg[0,0]=0
    beta=np.linalg.solve(xx.T@xx+reg,xx.T@y)
    residual=y-xx@beta
    # Partial lognormal bias correction targets typical rather than mean tail states.
    correction=float(np.mean(np.exp(residual)))
    artifacts[f'{asset}_{step}']=dict(center=center.tolist(),sd=sd.tolist(),beta=beta.tolist(),correction=correction)
    metrics.append(dict(asset=asset,step=step,n=len(X),log_rmse=float(np.std(residual)),correction=correction))
output=os.environ.get('HAR_FILE','har.json')
Path('agent',output).write_text(json.dumps(dict(fit_end=fit_end,models=artifacts),indent=2))
pd.DataFrame(metrics).to_csv('agent/results/'+Path(output).stem+'_fit.csv',index=False)
print(pd.DataFrame(metrics).to_string(index=False))
