"""Pre-validation seasonal profiles using absolute or squared five-minute returns."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient,ensure_workspace_cwd
ensure_workspace_cwd()
assets=json.loads(Path('agent/seasonality.json').read_text())['tables']
fit_end=os.environ.get('FIT_END','2026-07-31')
fit_start=os.environ.get('FIT_START','2026-06-01')
prefix=os.environ.get('ARTIFACT_PREFIX','')
variants={(kind,bins):{} for kind in ['abs','var'] for bins in [24,48,96]}
for asset in assets:
 df=load_lenient(asset,pd.Timestamp(fit_start,tz='UTC'),pd.Timestamp(fit_end+' 23:59',tz='UTC'))
 close=df.close.resample('5min').last();r=np.log(close).diff()
 for (kind,bins),tables in variants.items():
  measure=r.abs() if kind=='abs' else r.pow(2).clip(upper=r.pow(2).quantile(.999))
  total=measure.mean()
  bucket=(close.index.hour*60+close.index.minute)//(1440//bins)
  dow=close.index.dayofweek
  key=dow*bins+bucket
  grouped=measure.groupby(key).agg(['mean','count']).reindex(range(7*bins))
  groupkey=(dow>=5).astype(int)*bins+bucket
  prior=measure.groupby(groupkey).mean()/total
  # Prior pools weekdays; smooth local bins to regularize sparse weekend behavior.
  p=np.array([prior.get((i//bins>=5)*bins+i%bins,1.) for i in range(7*bins)])
  raw=(grouped['mean']/total).fillna(1).values;n=grouped['count'].fillna(0).values
  shape=(raw*n+p*120*24/bins)/(n+120*24/bins)
  if kind=='abs':shape=shape**2
  shape=np.clip(shape,.01,25);shape/=np.mean(shape)
  tables[asset]=shape.tolist()
for (kind,bins),tables in variants.items():
 Path(f'agent/{prefix}seasonality_{kind}{bins}.json').write_text(json.dumps(dict(fit_start=fit_start,fit_end=fit_end,bins=bins,tables=tables),indent=2))
print('Wrote',len(variants),'frozen seasonal profiles')
