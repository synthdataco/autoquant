"""Mean-absolute-scale preserving distribution ablations over learned forecasts."""
from __future__ import annotations
from math import gamma,pi,sqrt
import os
import numpy as np
from scipy.special import ndtri
from ml_models import scale_adjustment
from seasonal_models import forecast_variance

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','combo_mix40')

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,'vol',1.)
    steps=time_length//time_increment+1
    if VARIANT in ('combo_t3','combo_t5','combo_t8'):
        df=int(VARIANT[7:])
        t_abs=2*sqrt(df)*gamma((df+1)/2)/((df-1)*sqrt(pi)*gamma(df/2))
        z=rng.standard_t(df,size=(num_simulations,steps))*sqrt(2/pi)/t_abs
    elif VARIANT=='combo_antithetic':
        raw=rng.normal(size=((num_simulations+1)//2,steps))
        z=np.concatenate([raw,-raw],axis=0)[:num_simulations]
    elif 'lhs' in VARIANT:
        quantiles=ndtri((np.arange(num_simulations)+.5)/num_simulations)
        z=rng.permuted(np.broadcast_to(quantiles[:,None],(num_simulations,steps)),axis=0)
    else:
        z=rng.normal(size=(num_simulations,steps))
    if 'mix' in VARIANT:
        eta=int(VARIANT.split('mix')[1].split('_')[0])/100
        factors=rng.permutation(quantiles) if 'lhs' in VARIANT else rng.normal(size=num_simulations)
        z*=np.exp(eta*factors-.5*eta*eta)[:,None]
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*adjustment
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
