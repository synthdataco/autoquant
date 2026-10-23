"""Confirm the optional cached-prompt mode matches ordinary official backtest."""
import json
from pathlib import Path
import numpy as np,pandas as pd
for asset,fmt in [('BTC','24h'),('BTC','1h'),('AAPLX','24h')]:
 name=f'{asset}_{fmt}_2026-09-16'
 x=pd.read_parquet(Path('agent/results/parity')/(name+'.parquet')).sort_values('start_time')
 y=pd.read_parquet(Path('agent/results/seasonal')/(name+'.parquet')).sort_values('start_time')
 assert x.start_time.to_list()==y.start_time.to_list()
 np.testing.assert_allclose(x.crps.to_numpy(),y.crps.to_numpy(),rtol=0,atol=1e-10)
 difference=float(np.max(np.abs(x.crps.to_numpy()-y.crps.to_numpy())))
 print(asset,fmt,len(x),'maximum CRPS difference',difference,flush=True)
