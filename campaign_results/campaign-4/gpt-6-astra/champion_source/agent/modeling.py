"""Causal seasonal HAR volatility with a lognormal mixture of price paths.

The model uses only the supplied context and dated, locally stored training artifacts.
See report.md for official CRPS measurements and rejected alternatives.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_ARTIFACTS = json.loads(Path(__file__).with_name('model_artifacts.json').read_text())
_CHECKPOINTS = _ARTIFACTS['checkpoints']
_HOURS = (.25, 1, 6, 24, 72, 168)
_BINS = _ARTIFACTS['bins_per_day']


def _utc_time(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.Timestamp(value, unit='s', tz='UTC')
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')


def _checkpoint(start, time_length):
    # A minute labelled 23:59 is known only at the following midnight. The
    # artifact must also predate issuance, not merely the scored path start.
    available = start - pd.Timedelta(seconds={3600: 60, 86400: 120}.get(time_length, 0))
    eligible = [c for c in _CHECKPOINTS
                if pd.Timestamp(c['fit_end'], tz='UTC') + pd.Timedelta(days=1) <= available]
    return eligible[-1] if eligible else None


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    """Return metadata and positive paths beginning at the last observed close.

    Missing observations remain missing. An entirely unobserved context cannot
    define a starting price and raises a clear error rather than inventing one.
    """
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '42'))
    rng = np.random.default_rng(seed)
    if time_increment <= 0 or time_length < 0 or num_simulations <= 0:
        raise ValueError('positive increment/path count and nonnegative horizon required')
    if context_prices is None or 'close' not in context_prices:
        raise ValueError('context_prices must contain close prices')
    start = _utc_time(start_time)
    raw = pd.to_numeric(context_prices['close'], errors='coerce').astype(float)
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError('context_prices must have a minute DatetimeIndex')
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()
    raw = raw.where(np.isfinite(raw) & (raw > 0))
    observed = raw.dropna()
    if observed.empty:
        raise ValueError('context_prices needs at least one positive observed close')
    last = float(observed.iloc[-1])
    steps = time_length // time_increment
    if steps == 0:
        return (start_time, time_increment, *[[last] for _ in range(num_simulations)])

    checkpoint = _checkpoint(start, time_length)
    shape = np.ones(7 * _BINS)
    if checkpoint is not None:
        shape = np.asarray(checkpoint['seasonality'].get(asset, shape), dtype=float)

    def seasonal_indices(index):
        return index.dayofweek * _BINS + (index.hour * 60 + index.minute) // (1440 // _BINS)

    grid = raw.resample(f'{time_increment}s').last()
    normalized_abs = np.log(grid).diff().abs() / np.sqrt(shape[seasonal_indices(grid.index)])
    usable = normalized_abs.dropna()
    usable = usable[usable > 0]
    typical = float(usable.mean()) if len(usable) else 1e-5
    forecast = typical
    key = f'{asset}_{time_increment}'
    if checkpoint is not None and key in checkpoint['har']:
        fitted = checkpoint['har'][key]
        features = np.array([
            normalized_abs.iloc[-max(1, int(h * 3600 / time_increment)):].mean()
            for h in _HOURS
        ], dtype=float)
        # Missing feature summaries use a longer observed scale; no price is filled.
        fallback = float(normalized_abs.mean())
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = float(np.exp(fitted['center'][-1]))
        features = np.where(np.isfinite(features) & (features > 0), features, fallback)
        standardized = (np.log(np.maximum(features, 1e-10)) - fitted['center']) / fitted['sd']
        forecast = float(np.exp(np.dot(np.r_[1., standardized], fitted['beta'])) * fitted['correction'])
    if not np.isfinite(forecast) or forecast <= 0:
        forecast = typical

    future = pd.date_range(start, periods=steps, freq=f'{time_increment}s')
    level = (forecast * np.sqrt(np.pi / 2)) ** 2
    sigma = np.sqrt(np.maximum(level * shape[seasonal_indices(future)], 1e-14))
    half = (num_simulations + 1) // 2
    innovations = rng.standard_normal((half, steps))
    innovations = np.concatenate([innovations, -innovations], axis=0)[:num_simulations]
    spread = float(_ARTIFACTS['path_log_scale_sd'])
    factors = np.exp(spread * rng.standard_normal((half, 1)) - .5 * spread ** 2)
    factors = np.concatenate([factors, factors], axis=0)[:num_simulations]
    changes = innovations * factors * sigma
    logs = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(changes, axis=1)], axis=1)
    paths = last * np.exp(logs)
    return (start_time, time_increment, *paths.tolist())
