"""Symmetric, context-derived innovation laws with a learned forecast scale."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.special import ndtr

from ml_features import finite_ewm
from ml_models import scale_adjustment
from seasonal_models import forecast_variance, get_profile, seasonal_values

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'empirical_standard')


def empirical_quantiles(asset, start, step, context, uniforms, standardize=True):
    # A regular index makes absent rows explicit NaNs. No prices are filled.
    index = pd.date_range(context.index[0], context.index[-1], freq='1min')
    frame = context.reindex(index)
    p = np.asarray(frame.close, dtype=float)
    good = np.isfinite(p) & (p > 0)
    logp = np.full(len(p), np.nan)
    np.log(p, out=logp, where=good)
    k = step//60
    r = np.r_[np.full(k, np.nan), logp[k:]-logp[:-k]]
    complete = pd.Series(good.astype(float)).rolling(k+1, min_periods=k+1).sum().to_numpy() == k+1
    r[~complete] = np.nan
    table = get_profile(asset, str(context.index[-1].date()), 'America/New_York')
    season = seasonal_values(index, table, 'America/New_York')
    r /= np.sqrt(pd.Series(season).rolling(k, min_periods=k).sum().to_numpy())
    phase = pd.Timestamp(start).minute % k
    select = np.flatnonzero(np.asarray(index.minute) % k == phase)
    values = r[select]
    if standardize:
        prior = np.sqrt(finite_ewm(values*values, 720/k, len(values)))
        prior = np.r_[np.nan, prior[:-1]]
        values = values/np.maximum(prior, 1e-7)
        values = np.clip(values, -15., 15.)
    ages = np.asarray((index[-1]-index[select]).total_seconds())/60
    use = np.isfinite(values)
    values, ages = values[use], ages[use]
    if len(values) < 60:
        return None
    weights = np.exp2(-ages/4320)
    # Symmetry keeps the conditional center at zero and preserves the atom
    # from real flat candles. This is a distribution, not a price-gap repair.
    values = np.r_[values, -values]
    weights = np.tile(weights, 2)
    mean_abs = np.sum(weights*np.abs(values))/weights.sum()
    if mean_abs < 1e-12:
        return None
    values *= np.sqrt(2/np.pi)/mean_abs
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    coordinates = (np.cumsum(weights)-.5*weights)/weights.sum()
    return np.interp(uniforms, coordinates, values)


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    adjustment = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
    normal = rng.normal(size=(num_simulations, len(stepvar)+1))
    z = empirical_quantiles(asset, start_time, time_increment, context_prices,
                            ndtr(normal), standardize=VARIANT != 'empirical_raw')
    if z is None:
        z = normal
    if VARIANT == 'empirical_mix':
        eta = .4
        z *= np.exp(eta*rng.normal(size=num_simulations)-.5*eta*eta)[:, None]
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*adjustment
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
