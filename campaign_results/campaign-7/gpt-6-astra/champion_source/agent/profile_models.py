"""Profile resolution, robustness and weekday pooling over a fixed ML correction."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from seasonal_models import seasonal_values
from ml_models import scale_adjustment
from vol_models import valid_returns,ewma

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','prof_abs')

@lru_cache(maxsize=8)
def load_artifact(filename):
    return json.loads((Path(__file__).resolve().parent/'artifacts'/filename).read_text())

@lru_cache(maxsize=1024)
def table_for(asset,asof,variant):
    filename={'prof_abs':'seasonality_abs.json','prof_5m':'seasonality_5m.json','prof_abs5m':'seasonality_abs5m.json',
              'prof_abs5m60':'seasonality_abs5m60.json','prof_abs5m240':'seasonality_abs5m240.json'}.get(variant,'seasonality.json')
    data=load_artifact(filename)
    fits=data['assets'].get(asset,{})
    dates=[d for d in fits if d<=asof]
    if not dates:
        return np.ones((7,data['bins']))
    fit=fits[max(dates)]['clocks']['America/New_York']
    raw=np.asarray(fit['raw'],dtype=float)
    counts=np.asarray(fit['counts'],dtype=float)
    individual={'prof_raw70':.7,'prof_raw100':1.}.get(variant,.35)
    base=raw.copy()
    groups=[slice(0,4)] if variant=='prof_week' else [slice(0,5),slice(5,7)]
    for days in groups:
        num=(raw[days]*counts[days]).sum(axis=0)
        den=counts[days].sum(axis=0)
        common=np.divide(num,den,out=np.ones_like(num),where=den>0)
        base[days]=individual*raw[days]+(1-individual)*common
    flat=base.ravel()
    flat=.8*flat+.1*np.roll(flat,1)+.1*np.roll(flat,-1)
    flat=np.clip(flat,.015,20)
    return (flat/flat.mean()).reshape(raw.shape)

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,r,age=valid_returns(context_prices)
    tab=table_for(asset,str(context_prices.index[-1].date()),VARIANT)
    r=r/np.sqrt(seasonal_values(context_prices.index[1:],tab,'America/New_York'))
    h1,h2=(60,720) if time_length<=3600 else (720,4320)
    var=.75*ewma(r,age,h1)+.25*ewma(r,age,h2)
    future=pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1),periods=time_length//60,freq='1min')
    seasonal=seasonal_values(future,tab,'America/New_York')
    stepvar=max(var,1e-14)*seasonal.reshape(-1,time_increment//60).sum(axis=1)
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
    z=rng.normal(size=(num_simulations,time_length//time_increment+1))
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*adjustment
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
