"""Distinguish a learned forecast from simple robust or global scale calibration."""
from __future__ import annotations
import os
import numpy as np
from seasonal_models import forecast_variance

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','season_abs')

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    robust=1.0 if VARIANT=='season_abs' else (.75 if VARIANT=='season_abs75' else 0.)
    scalar=int(VARIANT[5:])/100 if VARIANT.startswith('scale') else 1.
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices,robust=robust)
    z=rng.normal(size=(num_simulations,time_length//time_increment+1))
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*scalar
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
