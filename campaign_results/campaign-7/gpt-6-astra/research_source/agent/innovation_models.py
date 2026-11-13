"""Innovation and stochastic-volatility ablations with fixed seasonal scale."""
from __future__ import annotations
import os
import numpy as np
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT','mix40')


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    steps=time_length//time_increment+1
    if VARIANT in ('t3','t5','t8'):
        df=int(VARIANT[1:])
        shocks=rng.standard_t(df,size=(num_simulations,steps))*np.sqrt((df-2)/df)
    else:
        shocks=rng.normal(size=(num_simulations,steps))
    if VARIANT.startswith('mix'):
        eta=int(VARIANT[3:])/100
        factor=np.exp(eta*rng.normal(size=num_simulations)-eta*eta)
        shocks*=factor[:,None]
    if VARIANT.startswith('sv'):
        eta=int(VARIANT[2:])/100
        half=30 if time_length<=3600 else 180
        rho=np.exp(-np.log(2)*time_increment/60/half)
        logscale=eta*rng.normal(size=num_simulations)
        z=rng.normal(size=(num_simulations,steps))
        innovation=eta*np.sqrt(1-rho*rho)
        for k in range(1,steps):
            logscale=rho*logscale+innovation*z[:,k]
            shocks[:,k]*=np.exp(logscale-eta*eta)
    shocks[:,0]=0
    shocks[:,1:]*=np.sqrt(stepvar)
    paths=last*np.exp(np.cumsum(shocks,axis=1))
    return (start_time,time_increment,*paths.tolist())
