"""Offline Synth fan: calendar variance, weekend HAR, and calibrated t shocks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


_SEASONALITY = json.loads((Path(__file__).parent / "seasonality.json").read_text())
_SEASONALITY_AUG = json.loads((Path(__file__).parent / "seasonality_aug31.json").read_text())
_HALFHOUR = json.loads((Path(__file__).parent / "seasonality_halfhour_aug31.json").read_text())
_QUARTERHOUR = json.loads((Path(__file__).parent / "seasonality_quarterhour_aug31.json").read_text())
_FIVEMINUTE = json.loads((Path(__file__).parent / "seasonality_fiveminute_aug31.json").read_text())
_HAR = json.loads((Path(__file__).parent / "har_1h.json").read_text())
_COM_EQU = {"XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"}


def _factors(asset: str, times: pd.DatetimeIndex) -> np.ndarray:
    if asset in _COM_EQU:
        profile = _FIVEMINUTE[asset]
        weekday = np.asarray(profile["weekday"], dtype=float)
        weekend = np.asarray(profile["weekend"], dtype=float)
        bins = times.hour * 12 + times.minute // 5
        return np.where(times.dayofweek >= 5, weekend[times.hour], weekday[bins])
    profile = (_SEASONALITY_AUG if asset in _COM_EQU else _SEASONALITY).get(asset)
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
    if len(returns) > 10:
        adjusted_sq = returns * returns / historical_factors
        sigma2 = float(np.mean(adjusted_sq))
        if asset in {"BTC", "ETH", "SOL", "XRP", "HYPE"}:
            # Crypto volatility changes rapidly even after its weaker calendar
            # pattern is removed. Blend seven-day and recent levels.
            age_minutes = (times[-1] - times[1:][adjacent]) / np.timedelta64(1, "m")
            weights = np.exp2(-np.asarray(age_minutes, dtype=float) / 1440.0)
            recent_sigma2 = float(np.average(adjusted_sq, weights=weights))
            sigma2 = 0.5 * sigma2 + 0.5 * recent_sigma2
            if time_length == 3600 and time_increment == 60 and (pd.Timestamp(start_time).dayofweek >= 5 or asset == "HYPE") and asset in _HAR:
                profile = _HAR[asset]
                feature_values = []
                for width in profile["windows"]:
                    in_window = np.asarray(age_minutes, dtype=float) < width
                    if in_window.sum() < 0.8 * width:
                        break
                    feature_values.append(float(np.mean(adjusted_sq[in_window])))
                if len(feature_values) == len(profile["windows"]):
                    x = np.log(np.maximum(feature_values, 1e-12))
                    standardized = (x - np.asarray(profile["x_mean"])) / np.asarray(profile["x_sd"])
                    coefficient = np.asarray(profile["coef"])
                    sigma2 = float(profile["calibration"] * np.exp(coefficient[0] + standardized @ coefficient[1:]))
    else:
        sigma2 = 1e-6
    if asset in {"BTC", "ETH", "SOL", "XRP", "HYPE"} and time_length == 86_400 and time_increment == 300:
        sigma2 *= 0.70 if pd.Timestamp(start_time).dayofweek >= 5 else 0.85
    sigma2 = max(sigma2, 1e-12)

    steps = time_length // time_increment + 1
    minutes_per_step = time_increment // 60
    start = pd.Timestamp(start_time)
    future_times = pd.date_range(start + pd.Timedelta(minutes=1), periods=(steps - 1) * minutes_per_step, freq="min", tz="UTC")
    future_factors = _factors(asset, future_times).reshape(steps - 1, minutes_per_step).sum(axis=1)
    scales = np.sqrt(sigma2 * future_factors)
    if time_length == 3600 and time_increment == 60 and asset in {"ETH", "SOL", "XRP"}:
        shocks = rng.standard_t(df=3, size=(num_simulations, steps - 1)) * np.sqrt(1.0 / 3.0) * scales
    elif asset in _COM_EQU and time_length == 86_400 and time_increment == 300:
        first = rng.standard_t(df=5, size=((num_simulations + 1) // 2, steps - 1))
        shocks = np.concatenate((first, -first[:num_simulations // 2]), axis=0) * np.sqrt(3.0 / 5.0) * scales
    else:
        shocks = rng.standard_t(df=5, size=(num_simulations, steps - 1)) * np.sqrt(3.0 / 5.0) * scales
        if time_length == 3600 and time_increment == 60 and asset == "BTC":
            path_scale = np.exp(0.3 * rng.normal(size=(num_simulations, 1)) - 0.3**2)
            shocks *= path_scale
    log_paths = np.concatenate((np.zeros((num_simulations, 1)), np.cumsum(shocks, axis=1)), axis=1)
    paths = last * np.exp(log_paths)
    return (start_time, time_increment, *[path.tolist() for path in paths])
