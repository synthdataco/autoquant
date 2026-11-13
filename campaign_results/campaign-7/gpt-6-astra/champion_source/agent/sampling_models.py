"""Seeded low-discrepancy and bridge sampling for the same predictive process."""
from __future__ import annotations
from functools import lru_cache
import os
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','sobol_bridge')

@lru_cache(maxsize=8)
def normal_coordinates(seed,n,dimensions,ranked):
    engine=qmc.Sobol(d=dimensions,scramble=True,seed=seed)
    u=engine.random_base2(int(np.ceil(np.log2(max(n,1)))))[:n]
    if ranked and n>1:
        order=np.argsort(u,axis=0)
        ranks=np.empty_like(order)
        np.put_along_axis(ranks,order,np.arange(n)[:,None],axis=0)
        u=(ranks+.5)/n
    z=ndtri(np.clip(u,1e-12,1-1e-12))
    z.setflags(write=False)
    return z

def brownian_bridge(z,stepvar):
    n,steps=z.shape
    t=np.r_[0.,np.cumsum(stepvar)]
    values=np.zeros((n,steps+1))
    values[:,-1]=np.sqrt(t[-1])*z[:,0]
    queue=[(0,steps)]
    k=1
    while queue:
        left,right=queue.pop(0)
        if right-left<=1:
            continue
        mid=(left+right)//2
        span=max(t[right]-t[left],1e-30)
        a=(t[right]-t[mid])/span
        b=(t[mid]-t[left])/span
        variance=(t[mid]-t[left])*(t[right]-t[mid])/span
        values[:,mid]=a*values[:,left]+b*values[:,right]+np.sqrt(max(variance,0.))*z[:,k]
        k+=1
        queue.extend([(left,mid),(mid,right)])
    return values

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
    steps=time_length//time_increment
    mixture='mix' in VARIANT
    ranked='rank' in VARIANT
    z=normal_coordinates(seed,num_simulations,steps+int(mixture),ranked)
    if 'bridge' in VARIANT:
        logpaths=brownian_bridge(z[:,:steps],stepvar)
    else:
        logpaths=np.column_stack([np.zeros(num_simulations),np.cumsum(z[:,:steps]*np.sqrt(stepvar),axis=1)])
    if mixture:
        eta=.4
        logpaths*=np.exp(eta*z[:,-1]-.5*eta*eta)[:,None]
    logpaths*=adjustment
    paths=last*np.exp(logpaths)
    return (start_time,time_increment,*paths.tolist())
