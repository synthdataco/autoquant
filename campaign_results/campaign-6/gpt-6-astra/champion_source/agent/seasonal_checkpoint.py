"""Research simulator: seasonal EWMA variance with configurable innovation laws.

All forecasts use only the supplied context. Seasonal tables were fitted before validation.
Configurations are recorded with experiments; modeling.py will freeze the selected choices.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

@lru_cache(maxsize=4)
def table(name='seasonality.json'):
    return json.loads((Path(__file__).parent/name).read_text())['assets']

def shape_at(index, asset, seasonal, name='seasonality.json'):
    if not seasonal:return np.ones(len(index))
    local=index.tz_convert('America/New_York')
    arr=np.asarray(table(name)[asset])
    return arr[(local.dayofweek>=5).astype(int),local.hour*2+local.minute//30]

@lru_cache(maxsize=4)
def read_calendar(path):
    if not path or not Path(path).exists():return None
    return pd.read_parquet(path)

@lru_cache(maxsize=1)
def event_parameters():
    return json.loads((Path(__file__).parent/'event_multipliers.json').read_text())['groups']

def event_factor(index, asset, step_seconds, calendar_path, holidays=False):
    cal=read_calendar(calendar_path)
    factors=np.ones(len(index))
    if cal is None or cal.empty:return factors
    lower=index[0]-pd.Timedelta(hours=12)
    upper=index[-1]
    cal=cal[(cal.time_utc>=lower)&(cal.time_utc<=upper)]
    group='crypto' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'other'
    params=event_parameters()[group]
    for e in cal.itertuples():
        if e.scope!='all' and asset not in e.scope.split(','):continue
        kernel=params.get(e.event,{})
        if holidays and e.event in ['NYSE_CLOSED','NYSE_EARLY_CLOSE']:
            kernel={'0':.15}
        total=np.ones(len(index))
        for lo,mult in kernel.items():
            lo=int(lo);hi={0:5,5:30,30:120}[lo]
            if e.event.startswith('NYSE_'):hi=390 if e.event=='NYSE_CLOSED' else 180
            start=e.time_utc+pd.Timedelta(minutes=lo)
            stop=e.time_utc+pd.Timedelta(minutes=hi)
            # Exact overlap with the simulated return interval, including off-grid announcements.
            left=np.maximum(np.asarray((index-pd.Timedelta(seconds=step_seconds)-start).total_seconds()),0)
            right=np.minimum(np.asarray((index-start).total_seconds()),(stop-start).total_seconds())
            total+=(mult-1)*np.maximum(right-left,0)/step_seconds
        if e.event.startswith('NYSE_'):factors=np.minimum(factors,total)
        else:factors=np.maximum(factors,total)
    return factors

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
    rng=np.random.default_rng(seed)
    config={'seasonal': True}
    config['_calendar_path']=os.environ.get('SYNTH_BENCHMARK_CALENDAR','')
    return simulate_config(asset,start_time,time_increment,time_length,num_simulations,context_prices,rng,config)

def simulate_config(asset,start_time,time_increment,time_length,num_simulations,context_prices,rng,c):
    steps=time_length//time_increment
    short=time_length<=3600
    seasonal=c.get('seasonal',False)
    shape_name=c.get('table','seasonality.json')
    close=context_prices.close.astype(float)
    valid=close.dropna();last=float(valid.iloc[-1])
    # A lag of 1 minute for 1h and 5 minutes for 24h matches the base scored increments.
    lag=c.get('lag',1 if short else 5)
    r=np.log(close).diff(lag)
    good=np.isfinite(r.to_numpy())
    rv=r.to_numpy()[good]**2/lag
    idx=close.index[good]
    weights_shape=shape_at(idx,asset,seasonal,shape_name)
    rv=rv/weights_shape
    age=np.asarray((pd.Timestamp(start_time)-idx).total_seconds())/60
    if len(rv)<10:
        rv=np.array([1e-8]);age=np.array([1.0])
    def estimate(half):
        w=np.exp2(-age/half)
        return np.sum(w*rv)/max(np.sum(w),1e-12)
    if c.get('estimator','ewma')=='constant':
        initial=long=float(np.mean(rv))
    else:
        long=estimate(c.get('long_half',4320))
        if short:
            initial=.6*estimate(c.get('fast_half',60))+.25*estimate(360)+.15*long
        else:
            initial=.5*estimate(c.get('fast_half',720))+.5*long
    future=pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(seconds=time_increment),periods=steps,freq=pd.Timedelta(seconds=time_increment))
    sf=shape_at(future,asset,seasonal,shape_name)
    if c.get('events',False):
        sf*=event_factor(future,asset,time_increment,c.get('_calendar_path',''),c.get('holidays',False))
    decay=np.exp2(-np.arange(1,steps+1)*time_increment/60/c.get('revert_half',720 if not short else 240))
    variance=(long+(initial-long)*decay)*sf*time_increment/60
    scale=c.get('scale',1.0)
    if isinstance(scale,dict):scale=scale.get(asset,1.0)
    sig=np.sqrt(np.maximum(variance,1e-14))*scale
    n=num_simulations
    if c.get('antithetic',False):
        z=rng.normal(size=((n+1)//2,steps));z=np.concatenate([z,-z])[:n]
    else:z=rng.normal(size=(n,steps))
    df=c.get('df',0)
    if df:z=z/np.sqrt(rng.chisquare(df,size=z.shape)/df)*np.sqrt((df-2)/df)
    mix=c.get('mix',0)
    if mix:
        # Multiplicative path-level volatility uncertainty, E[scale^2]=1.
        z*=np.exp(mix*rng.normal(size=(n,1))-mix*mix)
    shocks=z*sig
    logs=np.column_stack([np.zeros(n),np.cumsum(shocks,axis=1)])
    paths=last*np.exp(np.clip(logs,-50,50))
    return (start_time,time_increment,*paths.tolist())
