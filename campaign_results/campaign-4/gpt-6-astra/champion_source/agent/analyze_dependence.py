import json
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
season=json.loads(Path('agent/seasonality.json').read_text())
rows=[];params={}
for asset in season['tables']:
 df=load_lenient(asset,pd.Timestamp('2026-06-01',tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
 for step in [60,300]:
  if step==60 and asset not in ['BTC','ETH','SOL','XRP','HYPE']:continue
  r=np.log(df.close.resample(f'{step}s').last()).diff()
  shape=np.array(season['tables'][asset]);r=r/np.sqrt(shape[r.index.dayofweek*24+r.index.hour])
  capped=r.clip(lower=r.quantile(.001),upper=r.quantile(.999))
  acf=float(capped.autocorr())
  ratio=float(capped.rolling(6).sum().var()/(6*capped.var()))
  rows.append(dict(asset=asset,step=step,acf1=acf,var6_ratio=ratio))
  params[f'{asset}_{step}']=float(np.clip(acf,-.3,.15))
Path('agent/dependence.json').write_text(json.dumps(dict(fit_end='2026-07-31',parameters=params),indent=2))
print(pd.DataFrame(rows).to_string(index=False))
