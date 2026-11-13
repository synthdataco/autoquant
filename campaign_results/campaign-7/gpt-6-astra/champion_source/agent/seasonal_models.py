"""Causal clock-seasonality ablations on top of the EWMA mixture."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from vol_models import valid_returns, ewma

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'season_ny')


@lru_cache(maxsize=1)
def artifact():
    path = Path(__file__).resolve().parent/'artifacts/seasonality.json'
    return json.loads(path.read_text()) if path.exists() else {'bins':96, 'assets':{}}


@lru_cache(maxsize=1024)
def get_profile(asset, asof, clock, strength=1.0, individual=0.35):
    data = artifact()
    fits = data['assets'].get(asset, {})
    keys = [key for key in fits if key <= asof]
    bins = data['bins']
    if not keys:
        return np.ones((7,bins))
    fit = fits[max(keys)]['clocks'][clock]
    raw = np.asarray(fit['raw'], dtype=float)
    counts = np.asarray(fit['counts'], dtype=float)
    base = raw.copy()
    for days in [slice(0,5),slice(5,7)]:
        numerator = (raw[days]*counts[days]).sum(axis=0)
        den = counts[days].sum(axis=0)
        common = np.divide(numerator,den,out=np.ones_like(numerator),where=den>0)
        base[days] = individual*raw[days] + (1-individual)*common
    # Gentle circular smoothing on the whole week, not across unrelated hours.
    flat = base.ravel()
    flat = .8*flat + .1*np.roll(flat,1) + .1*np.roll(flat,-1)
    flat = np.clip(flat, .015, 20)**strength
    return (flat/np.mean(flat)).reshape(7,bins)


def seasonal_values(index, table, clock):
    idx = pd.DatetimeIndex(index).tz_convert(clock)
    bins = table.shape[1]
    bucket = np.asarray((idx.hour*60+idx.minute)*bins//1440)
    return table[np.asarray(idx.day_of_week),bucket]


def forecast_variance(asset, start_time, time_increment, time_length, context,
                      clock='America/New_York', strength=1.0, robust=0.0):
    last,r,age = valid_returns(context)
    table = get_profile(asset, str(context.index[-1].date()), clock, strength)
    history = seasonal_values(context.index[1:],table,clock)
    r = r / np.sqrt(history)
    h1,h2 = (60,720) if time_length<=3600 else (720,4320)
    var = .75*ewma(r,age,h1)+.25*ewma(r,age,h2)
    if robust:
        avar = .75*ewma(r,age,h1,True)+.25*ewma(r,age,h2,True)
        var = (1-robust)*var + robust*avar
    future = pd.date_range(pd.Timestamp(start_time)+pd.Timedelta(minutes=1),periods=time_length//60,freq='1min')
    seasonal = seasonal_values(future,table,clock)
    stepvar = max(var,1e-14)*seasonal.reshape(-1,time_increment//60).sum(axis=1)
    return last,stepvar


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng = np.random.default_rng(seed)
    clock = {'season_utc':'UTC','season_ldn':'Europe/London'}.get(VARIANT,'America/New_York')
    strength = 0.0 if VARIANT=='season_none' else (0.7 if VARIANT=='season_shrink' else 1.0)
    robust = 0.5 if VARIANT=='season_robust' else 0.0
    last,stepvar = forecast_variance(asset,start_time,time_increment,time_length,context_prices,clock,strength,robust)
    steps = time_length//time_increment+1
    shocks = rng.normal(size=(num_simulations,steps))
    shocks[:,0] = 0
    shocks[:,1:] *= np.sqrt(stepvar)
    paths = last*np.exp(np.cumsum(shocks,axis=1))
    return (start_time,time_increment,*paths.tolist())
