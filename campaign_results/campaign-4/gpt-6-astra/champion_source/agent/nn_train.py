"""Conditional volatility distribution ablation: small neural residual over HAR.
Uses only pre-August targets and negative log-likelihood, never a private CRPS formula.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from predict import load_lenient,ensure_workspace_cwd
from vol_features import features,ASSETS
ensure_workspace_cwd();torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.35)
device='cuda' if torch.cuda.is_available() else 'cpu'
season=json.loads(Path('agent/seasonality.json').read_text());har=json.loads(Path('agent/har.json').read_text())
data={k:[] for k in ['crypto1','crypto24','com24']}
for asset in ASSETS:
 begin='2026-01-01' if asset in ASSETS[:5] else '2026-06-01'
 df=load_lenient(asset,pd.Timestamp(begin,tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
 for step in [60,300] if asset in ASSETS[:5] else [300]:
  x,a,base=features(df.close,asset,step,season)
  horizon=3600//step if step==60 else 86400//step
  target=a.shift(-1).iloc[::-1].rolling(horizon,min_periods=int(horizon*.75)).mean().iloc[::-1]
  fit=har['models'][f'{asset}_{step}'];ls=np.column_stack([x[f'a{h}'].to_numpy()+x['base'].to_numpy() for h in [.25,1,6,24,72,168]])
  forecast=np.c_[np.ones(len(x)),(ls-fit['center'])/fit['sd']]@fit['beta']+np.log(fit['correction'])
  rows=x.copy();rows['target']=np.log(target.clip(lower=1e-10))-forecast
  rows=rows.iloc[168*3600//step:-horizon:3600//step].replace([np.inf,-np.inf],np.nan).dropna()
  key='crypto1' if step==60 else 'crypto24' if asset in ASSETS[:5] else 'com24';data[key].append(rows)
artifacts={}
for key,frames in data.items():
 d=pd.concat(frames);y=d.pop('target').to_numpy();X=d.to_numpy();center=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1)
 X=np.clip((X-center)/sd,-5,5).astype('float32');Y=y.astype('float32')
 xt=torch.tensor(X,device=device);yt=torch.tensor(Y,device=device)
 nets=[]
 for seed in [42,123,987]:
  torch.manual_seed(seed)
  net=nn.Sequential(nn.Linear(X.shape[1],32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,2)).to(device)
  nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
  opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
  for epoch in range(80):
   permutation=torch.randperm(len(X),device=device)
   for indices in permutation.split(512):
    out=net(xt[indices]);mean=.3*torch.tanh(out[:,0]);logsd=-1.2+.7*torch.tanh(out[:,1]);res=(yt[indices]-mean)/torch.exp(logsd)
    loss=(logsd+.5*res**2).mean()+.3*(mean**2).mean()
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  net.eval()
  with torch.no_grad():
   out=net(xt);mean=.3*torch.tanh(out[:,0]);rmse=float(((yt-mean)**2).mean().sqrt())
  nets.append({k:v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()})
  print(key,seed,'n',len(X),'training residual rmse',rmse,flush=True)
 artifacts[key]=dict(center=center.tolist(),sd=sd.tolist(),nets=nets)
Path('agent/nn.json').write_text(json.dumps(dict(fit_end='2026-07-31',models=artifacts)))
print('Trained on',device,flush=True)
