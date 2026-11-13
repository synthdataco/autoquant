"""Preserved invalid research sampler; do not serve this implementation.

Pairing separate Sobol engines by row creates strong finite-fan correlations.
Kept solely to reproduce the diagnostic in iteration 41.
"""
from __future__ import annotations
import os
import numpy as np
from candidate_models import candidate_forecast
from sampling_models import normal_coordinates, brownian_bridge

CONTEXT_DAYS = 7


def draw_temporal(seed, count, stepvar, step_seconds):
    steps = len(stepvar)
    # Retain the original price coordinates and persistent-scale coordinate.
    price_normals = normal_coordinates(seed, count, steps+1, True)
    logpaths = brownian_bridge(price_normals[:, :steps], stepvar)
    independent_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
    state_normals = normal_coordinates(independent_seed, count, steps+1, True)
    total_log_sd = .4
    persistent_sd = total_log_sd*np.sqrt(.5)
    transient_sd = total_log_sd*np.sqrt(.5)
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
    paths = last*np.exp(draw_temporal(seed, num_simulations, variance, time_increment))
    return (start_time, time_increment, *paths.tolist())
