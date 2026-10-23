"""Causal standardized-return diagnostics; no surrogate CRPS."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from research_model import shape_at
out={};quantiles={};robust={}
for asset in ['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']:
 files=[p for p in sorted(default_store_root(asset).glob('date=*.parquet')) if '2026-06-01'<=p.stem[5:]<'2026-09-01']
 d=pd.concat([pd.read_parquet(p,columns=['timestamp','close']) for p in files]).set_index('timestamp').sort_index()
 shape=shape_at(d.index,asset,True);out[asset]={};quantiles[asset]={};robust[asset]={}
 for lag in [1,5]:
  r=np.log(d.close).diff(lag)/np.sqrt(shape*lag)
  v=.6*r.pow(2).ewm(halflife=60).mean().shift(lag)+.4*r.pow(2).ewm(halflife=4320).mean().shift(lag)
  z=(r/np.sqrt(v)).iloc[10080:].dropna().to_numpy();z=z[np.isfinite(z)]
  z/=np.sqrt(np.mean(z*z))
  out[asset][lag]={'n':len(z),'kurtosis':float(np.mean(z**4)),'tail3':float(np.mean(np.abs(z)>3)), 'abs_quantiles':np.quantile(np.abs(z),[.5,.9,.99]).tolist()}
  robust[asset][lag]=float(1/np.mean(np.abs(z))**2)
  q=np.quantile(np.minimum(np.abs(z),8),(np.arange(2048)+.5)/2048)
  q/=np.sqrt(np.mean(q*q));quantiles[asset][lag]=q.tolist()
 print(asset,out[asset],flush=True)
Path('agent/results/innovation_diagnostics.json').write_text(json.dumps(out,indent=2))

Path("agent/innovations.json").write_text(json.dumps({"training_end":"2026-09-01","assets":quantiles}))

Path("agent/robust_factors.json").write_text(json.dumps({"training_end":"2026-09-01","assets":robust},indent=2))
