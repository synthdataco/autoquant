"""Weak daily return dependence with unchanged one-step marginal variance."""
import os
import numpy as np
from candidate_models import candidate_forecast
from integrated_models import draw_logpaths
from sampling_models import normal_coordinates, brownian_bridge

CONTEXT_DAYS = 7


def draw_correlated(seed, count, variance):
    steps = len(variance)
    normals = normal_coordinates(seed, count, steps+1, True)
    original = brownian_bridge(normals[:, :steps], variance)
    fraction = variance/max(float(variance.sum()), 1e-30)
    q = .25
    increments = (np.diff(original, axis=1) +
                  (np.sqrt(1+q)-1)*original[:, -1:]*fraction)
    increments /= np.sqrt(1+q*fraction)
    logpaths = np.column_stack([np.zeros(count), np.cumsum(increments, axis=1)])
    logpaths *= np.exp(.4*normals[:, -1]-.5*.4*.4)[:, None]
    return logpaths


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, variance = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context_prices, shape=True, events=True)
    logpaths = (draw_correlated(seed, num_simulations, variance) if time_length == 86400
                else draw_logpaths(seed, num_simulations, variance, time_increment, mixture=.4))
    paths = last*np.exp(logpaths)
    return (start_time, time_increment, *paths.tolist())
