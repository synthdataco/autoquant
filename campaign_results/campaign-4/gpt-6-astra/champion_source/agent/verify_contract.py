"""Meaningful serving checks: both grids, raw gaps, seed control, and path counts."""
import importlib.util,os
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
spec=importlib.util.spec_from_file_location('model',Path('agent/modeling.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ctx=load_lenient('BTC',pd.Timestamp('2026-08-01',tz='UTC'),pd.Timestamp('2026-08-07 23:57',tz='UTC'))
for horizon,step in [(3600,60),(86400,300)]:
 for count in [7,1000]:
  raw=ctx.copy();raw.iloc[::11,:]=np.nan
  os.environ['SYNTH_BENCHMARK_SEED']='321'
  a=np.asarray(m.simulate('BTC','2026-08-08T00:00:00Z',step,horizon,count,raw)[2:])
  b=np.asarray(m.simulate('BTC','2026-08-08T00:00:00Z',step,horizon,count,raw)[2:])
  assert a.shape==(count,horizon//step+1)
  assert np.isfinite(a).all() and (a>0).all()
  assert np.all(a[:,0]==raw.close.dropna().iloc[-1])
  assert np.array_equal(a,b)
  os.environ['SYNTH_BENCHMARK_SEED']='322'
  c=np.asarray(m.simulate('BTC','2026-08-08T00:00:00Z',step,horizon,count,raw)[2:])
  assert not np.array_equal(a,c)
  print('PASS',horizon,step,count)
