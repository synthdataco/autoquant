"""Allocate the predicted average scale over forecast time using current state."""
from __future__ import annotations
import os
import numpy as np
from seasonal_models import forecast_variance,get_profile,seasonal_values
from ml_models import scale_adjustment
from vol_models import valid_returns,ewma

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','curve_half')

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
    _,r,age=valid_returns(context_prices)
    tab=get_profile(asset,str(context_prices.index[-1].date()),'America/New_York')
    r=r/np.sqrt(seasonal_values(context_prices.index[1:],tab,'America/New_York'))
    short=time_length<=3600
    hfast,hslow,hanchor=(15,60,720) if short else (60,720,4320)
    current=.75*ewma(r,age,hfast,True)+.25*ewma(r,age,hslow,True)
    average=.75*ewma(r,age,hslow,True)+.25*ewma(r,age,hanchor,True)
    ratio=float(np.clip(np.sqrt(current/max(average,1e-14)),.3,3.))
    half=(30 if short else 360) if VARIANT=='curve_slow' else (15 if short else 120)
    strength=.5 if VARIANT=='curve_half' else 1.
    elapsed=(np.arange(len(stepvar))+.5)*time_increment/60
    decay=np.exp2(-elapsed/half)
    # Preserve the average de-seasonalized absolute-return scale forecast.
    curve=np.maximum(.25,1+strength*(ratio-1)*(decay-decay.mean()))
    curve/=curve.mean()
    z=rng.normal(size=(num_simulations,len(stepvar)+1))
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*adjustment*curve
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
