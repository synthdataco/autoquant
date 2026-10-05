"""Measure pre-validation seasonal variance; fit a frozen prior, never future data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from predict import load_lenient, ensure_workspace_cwd
ensure_workspace_cwd()
assets='BTC ETH SOL XRP HYPE XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500'.split()
tables={}; summary=[]
for asset in assets:
    df=load_lenient(asset,pd.Timestamp('2026-06-01',tz='UTC'),pd.Timestamp('2026-07-31 23:59',tz='UTC'))
    # Five-minute endpoints: last actually traded close in each block, NaN blocks retained.
    close=df.close.resample('5min').last()
    r=np.log(close).diff()
    r2=r.pow(2)
    # Cap extreme isolated microstructure errors, retaining most tail variance.
    r2=r2.clip(upper=r2.quantile(.999))
    keys=close.index.dayofweek*24+close.index.hour
    stats=r2.groupby(keys).agg(['mean','count']).reindex(range(168))
    total=r2.mean()
    hourly=r2.groupby(close.index.hour).mean()/total
    weekend=r2.groupby(close.index.dayofweek>=5).mean()/total
    # Shrink each day/hour toward same hour with weekday/weekend distinction.
    groupkey=(close.index.dayofweek>=5).astype(int)*24+close.index.hour
    group=r2.groupby(groupkey).mean()/total
    prior=np.array([group.get((i//24>=5)*24+i%24,1.) for i in range(168)])
    raw=(stats['mean']/total).fillna(1).values
    n=stats['count'].fillna(0).values
    shape=(raw*n+prior*120)/(n+120)
    shape=np.clip(shape,.02,15);shape/=np.mean(shape)
    tables[asset]=shape.tolist()
    summary.append(dict(asset=asset,weekday_weekend_variance=float(weekend[False]/weekend[True]),hour_max_min_variance=float(hourly.max()/hourly.min()),five_min_sigma=float(np.sqrt(total)),coverage=float(close.notna().mean())))
Path('agent/seasonality.json').write_text(json.dumps(dict(fit_start='2026-06-01',fit_end='2026-07-31',tables=tables),indent=2))
pd.DataFrame(summary).to_csv('agent/results/seasonality.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
