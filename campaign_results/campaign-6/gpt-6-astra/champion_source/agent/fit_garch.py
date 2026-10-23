"""Fit two stationary GARCH coefficients; all CRPS tests remain official."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.optimize import minimize
from synth_lib.preparation.config import default_store_root
from garch_vol import prepare,filter_state,finite_ewma
assets=['BTC','ETH','SOL','XRP','HYPE'];out={'train_end_exclusive':'2026-09-01','assets':{},'diagnostics':{}}
for asset in assets:
 files=[p for p in sorted(default_store_root(asset).glob('date=*.parquet')) if '2026-05-25'<=p.stem[5:]<'2026-09-01']
 frame=pd.concat([pd.read_parquet(p,columns=['timestamp','close']) for p in files]).set_index('timestamp').sort_index()
 for phase,table,end in [('diagnostic','seasonality_jul01.json','2026-08-01'),('final','seasonality.json','2026-09-01')]:
  index,y,level,z=prepare(frame,asset,table)
  mask=np.asarray((index>=pd.Timestamp('2026-06-01',tz='UTC'))&(index<pd.Timestamp(end,tz='UTC')))
  def objective(theta):
   persistence,fraction=theta
   return filter_state(z,persistence*fraction,persistence,mask)[1]
  fits=[minimize(objective,start,method='L-BFGS-B',bounds=[(.90,.999),(.001,.30)],options={'maxiter':80,'ftol':1e-10}) for start in [[.99,.05],[.997,.02]]]
  fit=min(fits,key=lambda f:f.fun);p,f=fit.x;values={'alpha':float(p*f),'persistence':float(p),'beta':float(p-p*f),'fit_nll':float(fit.fun),'converged':bool(fit.success)}
  if phase=='diagnostic':
   validation=np.asarray(index>=pd.Timestamp('2026-08-01',tz='UTC'))
   values['august_nll']=float(filter_state(z,p*f,p,validation)[1])
   fast=.6*finite_ewma(y,15)+.25*finite_ewma(y,360)+.15*finite_ewma(y,4320)
   ratio=np.concatenate([[1.],fast[:-1]/np.maximum(level[:-1],1e-14)])
   valid=validation&np.isfinite(z)
   values['august_fast15_nll']=float(np.mean(np.log(ratio[valid])+z[valid]/ratio[valid]))
   out['diagnostics'][asset]=values
  else:out['assets'][asset]=values
  print(asset,phase,values,flush=True)
Path('agent/garch.json').write_text(json.dumps(out,indent=2))
