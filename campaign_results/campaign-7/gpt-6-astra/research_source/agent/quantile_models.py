"""Forecast the conditional distribution of future scale, not only its center."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import xgboost as xgb
from ml_features import NAMES,one_features
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','quant_raw')
ART=Path(__file__).resolve().parent/'artifacts/quantile'

@lru_cache(maxsize=1)
def manifest():
    return json.loads((ART/'manifest.json').read_text())

@lru_cache(maxsize=16)
def model(filename):
    booster=xgb.Booster(params={'device':'cpu','nthread':1})
    booster.load_model(ART/filename)
    booster.set_param({'device':'cpu','nthread':1})
    return booster

def scale_quantiles(asset,start_time,horizon,context,u):
    meta=manifest()
    date=str(context.index[-1].date())
    fits=[f for f in meta['fits'] if f['time_length']==horizon and f['cutoff']<=date]
    if not fits:
        return np.ones(len(u))
    fit=max(fits,key=lambda f:f['cutoff'])
    x,_=one_features(asset,start_time,horizon,context)
    x=x[[NAMES.index(name) for name in meta['features']]].reshape(1,-1)
    x[~np.isfinite(x)]=np.nan
    q=np.maximum.accumulate(model(fit['file']).inplace_predict(x)[0])
    alpha=np.asarray(meta['quantiles'])
    # Extend the observed tail slopes modestly; do not turn a few rare labels
    # into arbitrarily large serving paths.
    extended=np.r_[q[0]-1.2*(q[1]-q[0]),q,q[-1]+1.2*(q[-1]-q[-2])]
    grid=np.r_[.001,alpha,.999]
    return np.exp(np.clip(np.interp(u,grid,extended),np.log(.15),np.log(5.)))

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    steps=time_length//time_increment+1
    z=rng.normal(size=(num_simulations,steps))
    u=rng.permutation((np.arange(num_simulations)+.5)/num_simulations)
    factors=scale_quantiles(asset,start_time,time_length,context_prices,u)
    if VARIANT in ('quant_mean','quant_mean_sv'):
        central=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
        factors*=central/factors.mean()
    if VARIANT in ('quant_sv','quant_mean_sv'):
        eta=.25
        rho=np.exp(-np.log(2)*time_increment/60/(15 if time_length<=3600 else 120))
        state=eta*rng.normal(size=num_simulations)
        noise=rng.normal(size=(num_simulations,steps))
        for j in range(1,steps):
            state=rho*state+eta*np.sqrt(1-rho*rho)*noise[:,j]
            z[:,j]*=np.exp(state-.5*eta*eta)
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)
    z*=factors[:,None]
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
