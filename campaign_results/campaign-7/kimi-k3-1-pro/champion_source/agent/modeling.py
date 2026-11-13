"""Event-aware probabilistic price paths calibrated from minute OHLCV context."""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

CONTEXT_DAYS = 14
VOL_HALFLIFE_MINUTES = 30.0
BLOCK_MINUTES = 15


def _returns(context_prices) -> pd.Series:
    closes = context_prices["close"].astype(float)
    closes = closes.where(np.isfinite(closes) & (closes > 0))
    return np.log(closes).diff()


def _fast_variance(rets: np.ndarray) -> float:
    if rets.size < 10:
        return 5e-4**2
    alpha = 1.0 - np.exp(np.log(0.5) / VOL_HALFLIFE_MINUTES)
    initial = rets[: min(rets.size, 1440)]
    level = float(np.mean(initial * initial))
    for ret in rets:
        level += alpha * (ret * ret - level)
    return max(level, 1e-12)


def _seasonal_sigmas(rets: pd.Series, start_time, time_increment: int, steps: int) -> np.ndarray:
    values = rets.to_numpy(dtype=float)
    valid = np.isfinite(values)
    values = values[valid]
    if values.size < 100:
        return np.full(steps, 5e-4 * np.sqrt(time_increment / 60.0))
    local = rets.index[valid].tz_convert("America/New_York")
    weekend = local.day_of_week >= 5
    bins = (local.hour * 60 + local.minute) // 30
    future = pd.to_datetime(start_time) + pd.to_timedelta(
        (np.arange(steps) + 0.5) * time_increment, unit="s"
    )
    future_local = future.tz_convert("America/New_York")
    future_weekend = future_local.day_of_week >= 5
    future_bins = (future_local.hour * 60 + future_local.minute) // 30
    global_variance = max(float(np.var(values)), 1e-12)
    sigmas = np.empty(steps)
    for is_weekend in (False, True):
        future_mask = future_weekend == is_weekend
        if not future_mask.any():
            continue
        history_mask = weekend == is_weekend
        count = int(history_mask.sum())
        if count < 300:
            history_mask = np.ones(values.size, dtype=bool)
            count = values.size
        sample = values[history_mask]
        sample_bins = bins[history_mask]
        variance = float(np.mean(sample * sample))
        variance = (count * variance + 720 * global_variance) / (count + 720)
        mean_absolute = max(float(np.mean(np.abs(sample))), 1e-8)
        profile = np.ones(48)
        for bucket in range(48):
            bucket_values = np.abs(sample[sample_bins == bucket])
            if bucket_values.size >= 20:
                profile[bucket] = max(float(np.mean(bucket_values)) / mean_absolute, 0.25)
        profile = np.clip(profile, 0.25, 3.0)
        profile /= np.sqrt(np.mean(profile * profile))
        sigmas[future_mask] = np.sqrt(variance * time_increment / 60.0) * profile[
            future_bins[future_mask]
        ]
    return sigmas


def _event_multipliers(asset: str, start_time, time_increment: int, steps: int) -> np.ndarray:
    multipliers = np.ones(steps)
    path = os.environ.get("SYNTH_BENCHMARK_CALENDAR")
    if not path:
        return multipliers
    try:
        events = pd.read_parquet(path, columns=["time_utc", "event", "scope"])
        events["time_utc"] = pd.to_datetime(events["time_utc"], utc=True)
        start = pd.Timestamp(start_time)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        scope = events["scope"].astype(str)
        relevant = events["event"].isin(["CPI", "NFP", "FOMC", "EARNINGS"])
        relevant &= (scope == "all") | scope.str.split(",").apply(lambda values: asset in values)
        finish = start + pd.Timedelta(seconds=steps * time_increment)
        events = events[relevant & (events["time_utc"] >= start) & (events["time_utc"] <= finish)]
        ends = start + pd.to_timedelta(np.arange(1, steps + 1) * time_increment, unit="s")
        starts = ends - pd.Timedelta(seconds=time_increment)
        profiles = {
            "CPI": [3.0, 1.7, 1.3],
            "NFP": [1.8, 2.0, 1.1],
            "FOMC": [2.0, 2.5, 1.75] + [1.4] * 10,
            "EARNINGS": [3.0, 4.0, 4.0] + [5.0] * 4 + [3.0] * 6,
        }
        for event in events.itertuples():
            containing = np.flatnonzero((starts < event.time_utc) & (ends >= event.time_utc))
            first = int(containing[0]) if containing.size else 0
            width = max(1, round(300 / time_increment))
            for offset, value in enumerate(profiles[event.event]):
                index = first + offset * width
                if index >= steps:
                    break
                multipliers[index : min(index + width, steps)] *= value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return np.ones(steps)
    return np.clip(multipliers, 1.0, 8.0)


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    seed = int(os.environ.get("SYNTH_BENCHMARK_SEED", "20261108"))
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if time_length == 3600:
        context_prices = context_prices.iloc[-7 * 1440 :]
    closes = context_prices["close"].to_numpy(dtype=float)
    valid = closes[np.isfinite(closes) & (closes > 0)]
    last = float(valid[-1]) if valid.size else 1.0
    steps = time_length // time_increment
    rets = _returns(context_prices)
    if time_length == 3600:
        minute_values = rets.dropna().to_numpy()
        sigma = np.sqrt(_fast_variance(minute_values) * time_increment / 60.0)
        sigmas = np.full(steps, sigma)
        sigmas *= _event_multipliers(asset, start_time, time_increment, steps)
        if minute_values.size >= 100:
            residual = minute_values / max(float(np.std(minute_values)), 1e-8)
            residual = np.clip(residual - np.median(residual), -8.0, 8.0)
            if residual.size > BLOCK_MINUTES:
                n_blocks = (steps + BLOCK_MINUTES - 1) // BLOCK_MINUTES
                starts = rng.integers(
                    0, residual.size - BLOCK_MINUTES + 1, size=(num_simulations, n_blocks)
                )
                indices = (starts[:, :, None] + np.arange(BLOCK_MINUTES)[None, None, :]).reshape(
                    num_simulations, n_blocks * BLOCK_MINUTES
                )[:, :steps]
                shocks = residual[indices] * sigmas
            else:
                shocks = residual[rng.integers(0, residual.size, size=(num_simulations, steps))] * sigmas
        else:
            shocks = rng.normal(size=(num_simulations, steps)) * sigmas
    else:
        sigmas = _seasonal_sigmas(rets, start_time, time_increment, steps)
        sigmas *= _event_multipliers(asset, start_time, time_increment, steps)
        clean = context_prices["close"].astype(float)
        clean = clean.where(np.isfinite(clean) & (clean > 0))
        five_minute = np.log(clean).diff(5).dropna().to_numpy()
        minute_values = rets.dropna().to_numpy()
        if five_minute.size >= 100 and minute_values.size >= 100:
            residual = five_minute / np.sqrt(5.0 * max(float(np.var(minute_values)), 1e-12))
            residual = np.clip(residual - np.median(residual), -8.0, 8.0)
            empirical = residual[rng.integers(0, residual.size, size=(num_simulations, steps))] * sigmas
            gaussian = rng.normal(size=(num_simulations, steps)) * sigmas
            shocks = np.where(rng.random((num_simulations, steps)) < 0.5, empirical, gaussian)
        else:
            shocks = rng.normal(size=(num_simulations, steps)) * sigmas
    half = num_simulations // 2
    shocks[half : 2 * half] = -shocks[:half]
    log_paths = np.concatenate(
        [np.zeros((num_simulations, 1)), np.cumsum(shocks, axis=1)], axis=1
    )
    paths = last * np.exp(log_paths)
    return (start_time, time_increment, *[p.tolist() for p in paths])
