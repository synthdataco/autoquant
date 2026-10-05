"""Causal range/activity features for the HAR ablation."""
import numpy as np
import pandas as pd
HOURS=[.25,1,6,24,72,168]
def features(frame,step,season,asset):
    grid=frame.resample(f'{step}s').agg(dict(close='last',high='max',low='min',volume='sum',trade_count='sum'))
    close=grid.close
    r=np.log(close).diff()
    bins=season.get('bins',24)
    shape=np.array(season['tables'][asset])
    norm=np.sqrt(shape[grid.index.dayofweek*bins+(grid.index.hour*60+grid.index.minute)//(1440//bins)])
    a=r.abs()/norm
    rg=np.log(grid.high/grid.low).where(grid.high>=grid.low)/norm
    f={}
    means={}
    for h in HOURS:
        n=max(1,int(h*3600/step))
        means[h]=a.rolling(n,min_periods=1).mean()
        f[f'abs{h}']=np.log(means[h].clip(lower=1e-10))
    for h in [1,6,24]:
        n=int(h*3600/step)
        rangemean=rg.rolling(n,min_periods=1).mean()
        f[f'range{h}']=np.log((rangemean/means[h]).clip(lower=.01,upper=100))
        for col in ['volume','trade_count']:
            short=grid[col].rolling(n,min_periods=1).mean()
            long=grid[col].rolling(168*3600//step,min_periods=1).mean()
            f[f'{col}{h}']=np.log((short/long).clip(lower=.01,upper=100)).fillna(0.)
    for h in [6,24]:
        n=int(h*3600/step)
        rms=np.sqrt((r/norm).pow(2).rolling(n,min_periods=1).mean())
        f[f'jump{h}']=np.log((rms/means[h]).clip(lower=.5,upper=10))
    return pd.DataFrame(f,index=grid.index),a
