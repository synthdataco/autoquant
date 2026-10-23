"""Finite-context sensitivity and warm latency; these are not CRPS results."""
import time,json
from pathlib import Path
import numpy as np,pandas as pd
from predict import load_minute_prices,WORKSPACE
from garch_vol import forecast,finite_ewma
rows=[]
for asset in ['BTC','ETH','SOL','XRP','HYPE']:
 t=pd.Timestamp('2026-09-15T12:00:00Z');frame=load_minute_prices(WORKSPACE/'market_data',asset,t-pd.Timedelta(days=14),t-pd.Timedelta(minutes=1))
 recent=frame.loc[t-pd.Timedelta(days=7):];future=pd.date_range(t+pd.Timedelta(minutes=1),periods=60,freq='min')
 before=time.perf_counter();a=forecast(recent,asset,future);latency=time.perf_counter()-before
 b=forecast(frame,asset,future);relative=float(np.max(np.abs(a/b-1)))
 row={'asset':asset,'max_relative_finite_context_difference':relative,'seconds':latency};rows.append(row);print(row,flush=True)
 if relative>.001:raise RuntimeError('Finite-context sensitivity exceeds preregistered diagnostic tolerance0.1%')
y=np.array([1.,np.nan,3.,4.,5.]);estimated=finite_ewma(y,half=3,window=3)
for i in range(len(y)):
 v=y[max(0,i-2):i+1];w=np.exp2(-np.arange(len(v)-1,-1,-1)/3);ok=np.isfinite(v)
 assert np.isclose(estimated[i],np.sum(w[ok]*v[ok])/np.sum(w[ok]),rtol=1e-12)
Path('agent/results/garch_context_verification.json').write_text(json.dumps(rows,indent=2))
