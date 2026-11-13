"""Clock- and context-informed zero atoms with a fixed continuous mixture."""
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri
from candidate_models import candidate_forecast
from sampling_models import normal_coordinates, brownian_bridge
from seasonal_models import seasonal_values

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'zero_full')


@lru_cache(maxsize=1)
def artifact():
    return json.loads((Path(__file__).resolve().parent/'artifacts/zero_mass.json').read_text())


@lru_cache(maxsize=1024)
def table_for(asset, step, asof):
    fits = artifact()['assets'].get(asset, {}).get(str(step), {})
    dates = [date for date in fits if date <= asof]
    return np.asarray(fits[max(dates)]['probability']) if dates else np.zeros((7, 96))


def probabilities(asset, start, step, horizon, context):
    table = table_for(asset, step, str(context.index[-1].date()))
    k = step//60
    future = pd.date_range(pd.Timestamp(start)+pd.Timedelta(seconds=step),
                           periods=horizon//step, freq=pd.Timedelta(seconds=step))
    predicted = seasonal_values(future, table, 'America/New_York')
    if not predicted.any():
        return predicted
    frame = context.reindex(pd.date_range(context.index[0], context.index[-1], freq='1min'))
    prices = frame.close.to_numpy(dtype=float)
    good = np.isfinite(prices) & (prices > 0)
    valid = pd.Series(good.astype(int)).rolling(k+1, min_periods=k+1).sum().to_numpy() == k+1
    flat = np.r_[np.zeros(k, dtype=bool), prices[k:] == prices[:-k]]
    phase = pd.Timestamp(start).minute % k
    valid &= np.asarray(frame.index.minute) % k == phase
    historical = seasonal_values(frame.index, table, 'America/New_York')
    age = np.asarray((frame.index[-1]-frame.index).total_seconds())/60
    weights = np.exp2(-age[valid]/4320)
    shift = 0.
    if len(weights) >= 60:
        actual = np.average(flat[valid], weights=weights)
        expected = np.average(historical[valid], weights=weights)
        shift = .5*(logit(np.clip(actual, .001, .99))-logit(np.clip(expected, .001, .99)))
        shift = np.clip(shift, -1., 1.)
    result = expit(logit(np.clip(predicted, .00001, .99999))+shift)
    result[predicted == 0] = 0.
    return np.clip(result, 0, .65)


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, variance = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context_prices, shape=True, events=True)
    probability = probabilities(asset, start_time, time_increment, time_length, context_prices)
    if VARIANT == 'zero_half':
        probability *= .5
    steps = len(variance)
    normals = normal_coordinates(seed, num_simulations, steps+1, True)
    original = brownian_bridge(normals[:, :steps], variance)
    standardized = np.diff(original, axis=1)/np.sqrt(variance)
    u = ndtr(standardized)
    lower = .5*(1-probability)
    upper = .5*(1+probability)
    atom = (u >= lower) & (u <= upper)
    continuous = np.where(u < lower, u/(1-probability), (u-probability)/(1-probability))
    z = ndtri(np.clip(continuous, 1e-12, 1-1e-12))/(1-probability)
    z[atom] = 0.
    logpaths = np.column_stack([np.zeros(num_simulations), np.cumsum(z*np.sqrt(variance), axis=1)])
    logpaths *= np.exp(.4*normals[:, -1]-.5*.4*.4)[:, None]
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
