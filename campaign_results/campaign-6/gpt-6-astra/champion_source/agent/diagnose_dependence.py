"""Return-dependence diagnostics only; all CRPS remains official."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
assets=['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500'];out={}
for asset in assets:
 fs=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if '2026-06-01'<=f.stem[5:]<'2026-09-01']
 d=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in fs]).set_index('timestamp').sort_index()
 price=np.log(d.close.astype(float));r={lag:price.diff(lag) for lag in [1,5,30,60]}
 moments={lag:float(x.pow(2).mean()/lag) for lag,x in r.items()}
 values={'rho1':float(r[1].corr(r[1].shift(1))),'rho5':float(r[5].corr(r[5].shift(5))),
         'vol5_over1':float(np.sqrt(moments[5]/moments[1])),
         'vol30_over5':float(np.sqrt(moments[30]/moments[5])),
         'vol60_over1':float(np.sqrt(moments[60]/moments[1]))}
 out[asset]=values;print(asset,{k:round(v,4) for k,v in values.items()},flush=True)
Path('agent/results/dependence_diagnostics.json').write_text(json.dumps(out,indent=2))
