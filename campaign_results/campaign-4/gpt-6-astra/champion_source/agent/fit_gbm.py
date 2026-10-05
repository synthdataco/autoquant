import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from predict import load_lenient,ensure_workspace_cwd
from vol_features import features,ASSETS
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality.json').read_text())
data={key:[] for key in ['crypto1','crypto24','com24']}
for asset in ASSETS:
    begin='2026-01-01' if asset in ASSETS[:5] else '2026-06-01'
    df=load_lenient(asset,pd.Timestamp(begin,tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
    for step in [60,300] if asset in ASSETS[:5] else [300]:
        x,a,base=features(df.close,asset,step,season)
        horizon=3600//step if step==60 else 86400//step
        target=a.shift(-1).iloc[::-1].rolling(horizon,min_periods=max(1,horizon//2)).mean().iloc[::-1]
        rows=x.copy();rows['y']=np.log((target/base).clip(lower=1e-6))
        rows=rows.iloc[168*3600//step:-horizon:3600//step]
        rows=rows.replace([np.inf,-np.inf],np.nan).dropna(subset=['y','base'])
        key='crypto1' if step==60 else 'crypto24' if asset in ASSETS[:5] else 'com24'
        data[key].append(rows)
artifacts={}
for key,frames in data.items():
    d=pd.concat(frames);y=d.pop('y').to_numpy();X=d.to_numpy()
    model=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10,learning_rate=.04,early_stopping=False,random_state=42,categorical_features=[X.shape[1]-1])
    model.fit(X,y);residual=y-model.predict(X)
    correction=float(np.mean(np.exp(residual)))
    artifacts[key]=dict(model=model,correction=correction)
    print(key,len(X),'training log residual sd',float(residual.std()),'bias correction',correction,flush=True)
with open('agent/gbm.pkl','wb') as f:pickle.dump(dict(fit_end='2026-07-31',models=artifacts),f)
