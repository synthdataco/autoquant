"""Serving champion v9: seasonal, event-aware, group-specific variance forecasts.

All forecasts use only the supplied context. Seasonal tables were fitted before validation.
Serving choices are fixed below and do not read research configuration from the environment.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from functools import lru_cache
import importlib.util
import numpy as np
import pandas as pd

@lru_cache(maxsize=1)
def _bridge_normals():
    spec = importlib.util.spec_from_file_location(
        "_synth_champion_qmc_bridge", Path(__file__).with_name("qmc_bridge.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bridge_normals

@lru_cache(maxsize=4)
def table(name='seasonality.json'):
    return json.loads((Path(__file__).parent/name).read_text())['assets']

@lru_cache(maxsize=16)
def table_end(name):
    metadata=json.loads((Path(__file__).parent/name).read_text())
    return pd.Timestamp(metadata['train_end_exclusive'],tz='UTC')

def shape_at(index, asset, seasonal, name='seasonality.json',shrink=.1):
    if not seasonal:return np.ones(len(index))
    local=index.tz_convert('America/New_York')
    arr=np.asarray(table(name)[asset])
    if shrink!=.1 and arr.shape[0]==2:
        raw=np.maximum((arr-.1)/.9,1e-5)
        arr=(1-shrink)*raw+shrink
    day=local.dayofweek if arr.shape[0]==7 else (local.dayofweek>=5).astype(int)
    per_hour=arr.shape[1]//24
    return arr[day,local.hour*per_hour+local.minute//(60//per_hour)]

def clock_at(index,asset,lag):
    values=table('clock.json').get(asset)
    if values is None:return np.ones(len(index))
    minute=np.asarray(index.minute)
    return np.asarray(values)[(minute[:,None]-np.arange(lag))%60].mean(axis=1)

@lru_cache(maxsize=4)
def _calendar_version(path,mtime,size):
    return pd.read_parquet(path)

def read_calendar(path):
    if not path:return None
    try:stat=Path(path).stat()
    except FileNotFoundError:return None
    return _calendar_version(path,stat.st_mtime_ns,stat.st_size)

@lru_cache(maxsize=2)
def event_parameters(fine=False,name=None):
    name=name or ('event_multipliers_fine.json' if fine else 'event_multipliers.json')
    return json.loads((Path(__file__).parent/name).read_text())['groups']

def event_factor(index, asset, step_seconds, calendar_path, holidays=False, fine=False,power=1.,exclude=(),parameters=None):
    cal=read_calendar(calendar_path)
    factors=np.ones(len(index))
    if cal is None or cal.empty:return factors
    lower=index[0]-pd.Timedelta(hours=12)
    upper=index[-1]
    cal=cal[(cal.time_utc>=lower)&(cal.time_utc<=upper)]
    group='crypto' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'other'
    params=event_parameters(fine,parameters)[group]
    for e in cal.itertuples():
        if e.event in exclude:continue
        if e.scope!='all' and asset not in e.scope.split(','):continue
        kernel=params.get(e.event,{})
        if holidays and e.event in ['NYSE_CLOSED','NYSE_EARLY_CLOSE']:
            kernel={'0':.15}
        total=np.ones(len(index))
        for lo,mult in kernel.items():
            if mult>1:mult=mult**power
            lo=int(lo);hi=({0:1,1:5,5:30,30:120} if fine else {0:5,5:30,30:120})[lo]
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

@lru_cache(maxsize=1)
def residual_parameters():
    return json.loads((Path(__file__).parent/'residual.json').read_text())['models']

def residual_features(frame,asset):
    close=frame.close.astype(float)
    ret=np.log(close).diff().to_numpy()
    season=shape_at(frame.index,asset,True)
    rv=ret**2/season
    age=np.asarray((frame.index[-1]-frame.index).total_seconds())/60
    def ew(x,half):
        x=np.asarray(x,dtype=float);ok=np.isfinite(x)
        w=np.exp2(-age[ok]/half)
        return float(np.sum(w*x[ok])/max(w.sum(),1e-12))
    v15,v60,vlong=(ew(rv,h) for h in [15,60,4320])
    eps=1e-16
    rangevar=np.log(frame.high.astype(float)/frame.low.astype(float)).to_numpy()**2/(4*np.log(2))/season
    volume=frame.volume.astype(float).to_numpy();trades=frame.trade_count.astype(float).to_numpy()
    vals={
        'fast_ratio':np.clip(np.log((v15+eps)/(v60+eps)),-3,3),
        'slow_ratio':np.clip(np.log((v60+eps)/(vlong+eps)),-3,3),
        'range_ratio':np.clip(np.log((ew(rangevar,60)+eps)/(v60+eps)),-3,3),
        'volume_ratio':np.clip(np.log((ew(volume,60)+1)/(ew(volume,1440)+1)),-3,3),
        'trade_ratio':np.clip(np.log((ew(trades,60)+1)/(ew(trades,1440)+1)),-3,3),
        'return_signed':np.clip(np.nansum(ret[-60:])/np.sqrt(60*v60*season[-1]+eps),-4,4),
    }
    return vals

CRYPTO={'BTC','ETH','SOL','XRP','HYPE'}
def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'));short=time_length<=3600
    config={'seasonal':True,'clock':True,'events':True,'event_power':1.5,'event_bar_alignment':short,
            'mix':.35 if short or asset not in CRYPTO else 0.,
            'fast_half':15 if short else 720,
            'seasonal_shrink':.02 if not short else .1,
            'scale':.95 if not short else 1.,
            'refresh':asset not in CRYPTO,
            'table':'seasonality_fine.json' if asset not in CRYPTO else 'seasonality.json',
            'refresh_table':'seasonality_oct01_fine.json',
            'refresh_schedule':['seasonality_sep14_fine.json','seasonality_oct05_fine.json'] if asset not in CRYPTO else [],
            'robust':.5 if asset not in CRYPTO else 0.,
            'sampler':'sobol_bridge' if short else 'sobol' if asset not in CRYPTO else 'random',
            '_calendar_path':os.environ.get('SYNTH_BENCHMARK_CALENDAR',''),'_seed':seed}
    return simulate_config(asset,start_time,time_increment,time_length,num_simulations,context_prices,np.random.default_rng(seed),config)

def simulate_config(asset,start_time,time_increment,time_length,num_simulations,context_prices,rng,c):
    steps=time_length//time_increment
    short=time_length<=3600
    seasonal=c.get('seasonal',False)
    shape_name=c.get('table','seasonality.json')
    if c.get('refresh',False):
        available=context_prices.index[-1]+pd.Timedelta(minutes=1)
        if available>=pd.Timestamp('2026-10-01',tz='UTC'):
            shape_name=c.get('refresh_table','seasonality_oct01.json')
    if c.get('refresh_schedule'):
        available=context_prices.index[-1]+pd.Timedelta(minutes=1)
        selected_end=table_end(shape_name)
        for name in c['refresh_schedule']:
            trained=table_end(name)
            if selected_end<trained<=available:
                shape_name=name;selected_end=trained
    close=context_prices.close.astype(float)
    valid=close.dropna();last=float(valid.iloc[-1])
    # A lag of 1 minute for 1h and 5 minutes for 24h matches the base scored increments.
    lag=c.get('lag',1 if short else 5)
    r=np.log(close).diff(lag)
    good=np.isfinite(r.to_numpy())
    rv=r.to_numpy()[good]**2/lag
    idx=close.index[good]
    weights_shape=shape_at(idx,asset,seasonal,shape_name,c.get('seasonal_shrink',.1))
    if c.get('clock',False):weights_shape*=clock_at(idx,asset,lag)
    rv=rv/weights_shape
    age=np.asarray((pd.Timestamp(start_time)-idx).total_seconds())/60
    if len(rv)<10:
        rv=np.array([1e-8]);age=np.array([1.0])
    def estimate(half):
        w=np.exp2(-age/half)
        raw=np.sum(w*rv)/max(np.sum(w),1e-12)
        blend=c.get('robust',0.)
        if blend:
            correction=float(table('robust_factors.json')[asset][str(lag)])
            absolute=(np.sum(w*np.sqrt(rv))/max(np.sum(w),1e-12))**2*correction
            return (1-blend)*raw+blend*absolute
        return raw
    if c.get('estimator','ewma')=='constant':
        initial=long=float(np.mean(rv))
    else:
        long=estimate(c.get('long_half',4320))
        if short:
            initial=.6*estimate(c.get('fast_half',60))+.25*estimate(360)+.15*long
        else:
            initial=.5*estimate(c.get('fast_half',720))+.5*long
    future=pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(seconds=time_increment),periods=steps,freq=pd.Timedelta(seconds=time_increment))
    sf=shape_at(future,asset,seasonal,shape_name,c.get('seasonal_shrink',.1))
    if c.get('clock',False):sf*=clock_at(future,asset,max(1,time_increment//60))
    if c.get('events',False):
        event_index=future+pd.Timedelta(seconds=60 if c.get('event_bar_alignment',False) else 0)
        sf*=event_factor(event_index,asset,time_increment,c.get('_calendar_path',''),c.get('holidays',False),c.get('event_fine',False),c.get('event_power',1.),c.get('event_exclude',()),c.get('event_table'))
    decay=np.exp2(-np.arange(1,steps+1)*time_increment/60/c.get('revert_half',720 if not short else 240))
    variance=(long+(initial-long)*decay)*sf*time_increment/60
    if c.get('chronos',False) or c.get('hourly_persistence',False):
        from chronos_vol import forecast_variance
        learned=forecast_variance(context_prices,asset,future,c['_seed'],c.get('hourly_persistence',False),c.get('chronos_model','chronos-t5-tiny'))
        base=np.maximum(long+(initial-long)*decay,1e-14)
        blend=c.get('chronos_blend',.25)
        variance*=1-blend+blend*np.clip(learned/base,.25,4.)
    scale=c.get('scale',1.0)
    if isinstance(scale,dict):scale=scale.get(asset,1.0)
    sig=np.sqrt(np.maximum(variance,1e-14))*scale
    if short and c.get('residual'):
        params=residual_parameters()[c['residual']]
        features=residual_features(context_prices,asset)
        x=np.array([features[k] for k in params['features']])
        standardized=np.nan_to_num((x-params['mean'])/np.asarray(params['std']))
        correction=float(standardized@np.asarray(params['coef'][1:]))
        sig*=np.exp(c.get('residual_strength',.5)*np.clip(correction,-.25,.25))
    n=num_simulations
    mixture_normal=None
    if c.get('sampler') in ['sobol','sobol_bridge']:
        from scipy.stats import qmc
        from scipy.special import ndtri
        engine=qmc.Sobol(d=steps+1,scramble=True,seed=int(rng.integers(0,2**32-1)))
        u=engine.random_base2(int(np.ceil(np.log2(max(n,1)))))[:n]
        normal=ndtri(np.clip(u,1e-12,1-1e-12))
        if c.get('sampler')=='sobol_bridge':
            bridge_normals = _bridge_normals()
            mixture_normal=normal[:,:1]
            z=bridge_normals(normal[:,1:],sig)
        else:z=normal[:,:steps];mixture_normal=normal[:,-1:]
    elif c.get('sampler')=='stratified':
        from scipy.special import ndtri
        ranks=rng.uniform(size=(n,steps)).argsort(axis=0)
        z=ndtri((ranks+.5)/n)
    elif c.get('antithetic',False) or c.get('paired',False):
        z=rng.normal(size=((n+1)//2,steps));z=np.concatenate([z,-z])[:n]
    else:z=rng.normal(size=(n,steps))
    if c.get('empirical',False):
        from scipy.special import ndtr
        quantiles=np.asarray(table('innovations.json')[asset][str(lag)])
        u=2*np.abs(ndtr(z)-.5)
        z=np.sign(z)*np.interp(u,(np.arange(len(quantiles))+.5)/len(quantiles),quantiles)
    df=c.get('df',0)
    if df:z=z/np.sqrt(rng.chisquare(df,size=z.shape)/df)*np.sqrt((df-2)/df)
    mix=c.get('mix',0)
    if mix and c.get('adaptive_mix',False):
        disagreement=np.log(max(initial,1e-14)/max(long,1e-14))
        mix=min(.55,float(np.sqrt(mix*mix+.10*disagreement*disagreement)))
    if c.get('mix_scope')=='non_crypto24h' and not short and asset in ['BTC','ETH','SOL','XRP','HYPE']:
        mix=0
    if mix:
        # Multiplicative path-level volatility uncertainty, E[scale^2]=1.
        if mixture_normal is None:
            if c.get('paired',False):
                half=rng.normal(size=((n+1)//2,1));mixture_normal=np.concatenate([half,half])[:n]
            else:mixture_normal=rng.normal(size=(n,1))
        if c.get('mix_revert',False):
            from scipy.signal import lfilter
            half_seconds=time_length*c.get('mix_half_fraction',.25)
            rho=2**(-time_increment/half_seconds)
            shocks=rng.normal(size=(n,steps))*np.sqrt(1-rho*rho)
            shocks[:,0]=mixture_normal[:,0]
            mixture_normal=lfilter([1.],[1.,-rho],shocks,axis=1)
        z*=np.exp(mix*mixture_normal-mix*mix)
    shocks=z*sig
    logs=np.column_stack([np.zeros(n),np.cumsum(shocks,axis=1)])
    paths=last*np.exp(np.clip(logs,-50,50))
    return (start_time,time_increment,*paths.tolist())
