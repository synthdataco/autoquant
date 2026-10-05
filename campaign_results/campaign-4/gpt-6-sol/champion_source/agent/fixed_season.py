"""Minute-volatility seasonality calibrated on the preceding seven days."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


_SEASONALITY = json.loads((Path(__file__).parent / "seasonality.json").read_text())


def _factors(asset: str, times: pd.DatetimeIndex) -> np.ndarray:
    profile = _SEASONALITY.get(asset)
    if profile is None:
        return np.ones(len(times), dtype=float)
    weekday = np.asarray(profile["weekday"], dtype=float)
    weekend = np.asarray(profile["weekend"], dtype=float)
    return np.where(times.dayofweek >= 5, weekend[times.hour], weekday[times.hour])


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    benchmark_seed = os.environ.get("SYNTH_BENCHMARK_SEED", "20260930")
    digest = hashlib.blake2b(f"{benchmark_seed}|{asset}|{start_time}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    closes = context_prices["close"].to_numpy(dtype=float)
    times = pd.DatetimeIndex(pd.to_datetime(context_prices.index, utc=True))
    valid = np.isfinite(closes) & (closes > 0)
    if not valid.any():
        raise ValueError(f"No usable context close for {asset}")
    last = float(closes[np.flatnonzero(valid)[-1]])
    adjacent = valid[1:] & valid[:-1] & (np.diff(times.to_numpy()) == np.timedelta64(1, "m"))
    log_closes = np.log(np.where(valid, closes, 1.0))
    returns = np.diff(log_closes)[adjacent]
    historical_factors = _factors(asset, times[1:])[adjacent]
    sigma2 = float(np.mean(returns * returns / historical_factors)) if len(returns) > 10 else 1e-6
    sigma2 = max(sigma2, 1e-12)

    steps = time_length // time_increment + 1
    minutes_per_step = time_increment // 60
    start = pd.Timestamp(start_time)
    future_times = pd.date_range(start + pd.Timedelta(minutes=1), periods=(steps - 1) * minutes_per_step, freq="min", tz="UTC")
    future_factors = _factors(asset, future_times).reshape(steps - 1, minutes_per_step).sum(axis=1)
    scales = np.sqrt(sigma2 * future_factors)
    shocks = rng.normal(size=(num_simulations, steps - 1)) * scales
    log_paths = np.concatenate((np.zeros((num_simulations, 1)), np.cumsum(shocks, axis=1)), axis=1)
    paths = last * np.exp(log_paths)
    return (start_time, time_increment, *[path.tolist() for path in paths])
