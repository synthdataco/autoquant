"""Pair opposite log-return paths at the same volatility scale."""
from __future__ import annotations

from functools import lru_cache
import os
import numpy as np
from scipy.special import ndtri

from candidate_models import candidate_forecast
from sampling_models import normal_coordinates, brownian_bridge

CONTEXT_DAYS = 7


@lru_cache(maxsize=8)
def antithetic_coordinates(seed, count, steps):
    pairs, extra = divmod(count, 2)
    primary_count = pairs+extra
    primary = normal_coordinates(seed, primary_count, steps+1, False)
    coordinates = np.concatenate([primary[:pairs, :steps], -primary[:pairs, :steps],
                                  primary[pairs:, :steps]], axis=0)
    if count > 1:
        order = np.argsort(coordinates, axis=0)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(count)[:, None], axis=0)
        coordinates = ndtri((ranks+.5)/count)
    scale_normal = primary[:, -1].copy()
    if primary_count > 1:
        order = np.argsort(scale_normal)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(primary_count)
        scale_normal = ndtri((ranks+.5)/primary_count)
    scales = np.concatenate([scale_normal[:pairs], scale_normal[:pairs], scale_normal[pairs:]])
    coordinates.setflags(write=False)
    scales.setflags(write=False)
    return coordinates, scales


def antithetic_logpaths(seed, count, stepvar, mixture=.4):
    z, scale_normal = antithetic_coordinates(seed, count, len(stepvar))
    paths = brownian_bridge(z, stepvar)
    paths *= np.exp(mixture*scale_normal-.5*mixture*mixture)[:, None]
    return paths


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    last, stepvar = candidate_forecast(asset, start_time, time_increment, time_length,
                                       context_prices, shape=True, events=True)
    paths = last*np.exp(antithetic_logpaths(seed, num_simulations, stepvar, .4))
    return (start_time, time_increment, *paths.tolist())
