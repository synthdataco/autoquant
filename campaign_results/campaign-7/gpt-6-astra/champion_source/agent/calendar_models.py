"""Scheduled-event effects over the causal learned volatility baseline."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from ml_models import scale_adjustment
from seasonal_models import forecast_variance,get_profile,seasonal_values
from vol_models import valid_returns,ewma

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','event_all')

@lru_cache(maxsize=1)
def coefficients():
    return json.loads((Path(__file__).resolve().parent/'artifacts/events.json').read_text())['fits']

@lru_cache(maxsize=4)
def read_calendar(path,version):
    return pd.read_parquet(path)

def calendar():
    # Resolve the production calendar inside each simulate call. The cache key
    # changes when the miner's daily refresh replaces or rewrites the file,
    # including refreshes that preserve the original modification time.
    value=os.environ.get('SYNTH_BENCHMARK_CALENDAR')
    if not value:
        return None
    path=Path(value)
    try:
        stat=path.stat()
        version=(stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_size)
        return read_calendar(str(path),version)
    except (OSError,ValueError):
        return None

def kind_of(names):
    for kind in ['EARNINGS','EIA_PETROLEUM','CPI','NFP','FOMC','PPI','PCE','GDP','RETAIL_SALES']:
        if kind in names:
            return 'GDP_PCE' if kind in ('GDP','PCE') else kind
    return None

def event_variance_factors(asset,index,horizon,asof,events,strength=1.,earnings=True,only_earnings=False,*,fitted=None):
    idx=pd.DatetimeIndex(index)
    factors=np.ones(len(idx))
    if events is None or len(idx)==0:
        return factors
    fits=coefficients() if fitted is None else fitted
    dates=[date for date in fits if date<=asof]
    if not dates:
        return factors
    rows=[r for r in fits[max(dates)] if r['asset']==asset and r['time_length']==horizon and (earnings or r['event']!='EARNINGS') and (not only_earnings or r['event']=='EARNINGS')]
    if not rows:
        return factors
    relevant=events[(events.time_utc>=idx[0]-pd.Timedelta(hours=8))&(events.time_utc<=idx[-1]+pd.Timedelta(minutes=30))]
    relevant=relevant[(relevant.scope=='all')|relevant.scope.str.split(',').map(lambda x:asset in x)]
    for t,group in relevant.groupby('time_utc'):
        kind=kind_of(set(group.event))
        for row in rows:
            if row['event']!=kind:
                continue
            low=t+pd.Timedelta(minutes=row['lo'])
            high=t+pd.Timedelta(minutes=row['hi'])
            mask=(idx>=low)&(idx<high)
            value=1.+strength*(row['amplitude']**2-1.)
            factors[mask]=np.maximum(factors[mask],value)
    return factors

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    events=calendar()
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
    asof=str(context_prices.index[-1].date())
    strength=.5 if VARIANT=='event_half' else 1.
    earnings=VARIANT!='event_macro'
    future=pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1),periods=time_length//60,freq='1min')
    table=get_profile(asset,asof,'America/New_York')
    seasonal=seasonal_values(future,table,'America/New_York')
    factors=event_variance_factors(asset,future,time_length,asof,events,strength,earnings)
    k=time_increment//60
    stepvar*=((seasonal*factors).reshape(-1,k).sum(axis=1)/seasonal.reshape(-1,k).sum(axis=1))
    if VARIANT in ('event_clean','event_clean_half','event_clean_earnings'):
        _,r,age=valid_returns(context_prices)
        hseason=seasonal_values(context_prices.index[1:],table,'America/New_York')
        normalized=r/np.sqrt(hseason)
        hist=event_variance_factors(asset,context_prices.index[1:],time_length,asof,events,strength,earnings,VARIANT=='event_clean_earnings')
        h1,h2=(60,720) if time_length<=3600 else (720,4320)
        old=.75*ewma(normalized,age,h1)+.25*ewma(normalized,age,h2)
        clean=.75*ewma(normalized/np.sqrt(hist),age,h1)+.25*ewma(normalized/np.sqrt(hist),age,h2)
        ratio=np.clip(clean/max(old,1e-14),.15,1.)
        stepvar*=np.sqrt(ratio) if VARIANT=='event_clean_half' else ratio
    steps=time_length//time_increment+1
    z=rng.normal(size=(num_simulations,steps))
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*adjustment
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
