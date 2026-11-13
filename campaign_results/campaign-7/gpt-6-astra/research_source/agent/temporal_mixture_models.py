"""Redistribute fixed log-scale uncertainty between a path and an OU state."""
from __future__ import annotations
import os
import numpy as np
from candidate_models import candidate_forecast
from sampling_models import normal_coordinates, brownian_bridge

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'temporal_state')


def draw_temporal(seed, count, stepvar, step_seconds, transient_fraction=.5):
    steps = len(stepvar)
    # One joint Sobol engine is essential: pairing separate scrambled engines
    # by row index creates strong finite-fan cross-correlations. The static
    # reference uses the identical joint coordinates to control sampling error.
    joint = normal_coordinates(seed, count, 2*(steps+1), True)
    price_normals = joint[:, :steps+1]
    logpaths = brownian_bridge(price_normals[:, :steps], stepvar)
    state_normals = joint[:, steps+1:]
    total_log_sd = .4
    if transient_fraction == 0:
        return logpaths*np.exp(total_log_sd*price_normals[:, -1]-.5*total_log_sd**2)[:, None]
    persistent_sd = total_log_sd*np.sqrt(1-transient_fraction)
    transient_sd = total_log_sd*np.sqrt(transient_fraction)
    half_life = 30 if steps*step_seconds <= 3600 else 180
    rho = np.exp2(-step_seconds/60/half_life)
    state = transient_sd*state_normals[:, 0]
    persistent = persistent_sd*price_normals[:, -1]
    increments = np.diff(logpaths, axis=1)
    for j in range(steps):
        state = rho*state + transient_sd*np.sqrt(1-rho*rho)*state_normals[:, j+1]
        increments[:, j] *= np.exp(persistent+state-.5*total_log_sd**2)
    return np.column_stack([np.zeros(count), np.cumsum(increments, axis=1)])


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, variance = candidate_forecast(asset, start_time, time_increment,
                                       time_length, context_prices,
                                       shape=True, events=True)
    fraction = 0. if VARIANT == 'temporal_static' else .5
    paths = last*np.exp(draw_temporal(seed, num_simulations, variance, time_increment, fraction))
    return (start_time, time_increment, *paths.tolist())
