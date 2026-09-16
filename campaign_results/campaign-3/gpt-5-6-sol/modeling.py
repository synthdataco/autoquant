"""Deterministic horizon-aware probabilistic price simulator.

The model combines native-step volatility, selective pooled HAR forecasts, calibrated
asset/horizon widths, and antithetic Gaussian or Student-t innovations.
Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length//time_increment + 1 points starting at the context's last price.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment + 1
    # Environment switches are used only by the campaign's controlled experiments.  The
    # defaults are the served model, so an offline harness does not need to set anything.
    default_window = 6 * 60 if time_length <= 3600 else 24 * 60
    window_minutes = int(os.environ.get("SYNTH_VOL_WINDOW", default_window))
    scale_24h = {
        "BTC": 0.70,
        "ETH": 0.90,
        "SOL": 0.90,
        "XRP": 0.90,
        "HYPE": 0.90,
        "XAU": 0.90,
        "NVDAX": 0.70,
        "GOOGLX": 0.70,
        "TSLAX": 0.70,
        "AAPLX": 0.70,
        "WTIOIL": 0.90,
        "SPCX": 0.50,
        "SP500": 0.90,
    }
    default_scale = 0.85 if time_length <= 3600 else scale_24h.get(asset, 0.70)
    sigma_scale = float(os.environ.get("SYNTH_SIGMA_SCALE", str(default_scale)))
    default_innovation = "normal" if time_length <= 3600 or asset == "BTC" else "student5"
    innovation = os.environ.get("SYNTH_INNOVATION", default_innovation)
    har_assets = {"SOL", "XAU", "SP500", "NVDAX", "GOOGLX", "TSLAX", "AAPLX"}
    default_estimator = "har" if time_length > 3600 and asset in har_assets else "std"
    vol_estimator = os.environ.get("SYNTH_VOL_ESTIMATOR", default_estimator)
    half_life = float(os.environ.get("SYNTH_HALF_LIFE", "360"))
    drift_lookback = int(os.environ.get("SYNTH_DRIFT_LOOKBACK", "0"))
    drift_coef = float(os.environ.get("SYNTH_DRIFT_COEF", "0"))
    ar1 = float(os.environ.get("SYNTH_AR1", "0"))
    martingale = os.environ.get("SYNTH_MARTINGALE", "0") == "1"
    if vol_estimator == "har":
        window_minutes = 7 * 24 * 60
    raw = context_prices["close"].iloc[-window_minutes:]
    closes = raw.dropna()
    last = float(closes.iloc[-1])
    # Do not turn a run of no-trade NaNs into a fictitious one-minute return.  Consecutive
    # observed rows alone are valid one-minute increments.
    values = np.asarray(raw, dtype=float)
    native_minutes = max(1, time_increment // 60)
    native_log_rets = np.log(values[native_minutes:]) - np.log(values[:-native_minutes])
    native_valid = np.isfinite(native_log_rets)
    rets = native_log_rets[native_valid]
    if rets.size > 10:
        # Estimate at the model's native step. This mirrors the validator's endpoint sampling:
        # a return is learned only when both endpoints are observed, while NaNs inside a 5m
        # interval are neither filled nor incorrectly treated as a longer one-minute move.
        if vol_estimator == "mad":
            center = float(np.median(rets))
            raw_sigma = 1.4826 * float(np.median(np.abs(rets - center)))
        elif vol_estimator == "ewma":
            # native_log_rets are produced once per minute (with a native-step lag), so the
            # half-life is in minutes even for the 5m format.
            ages = (native_log_rets.size - 1 - np.flatnonzero(native_valid)).astype(float)
            weights = np.exp2(-ages / max(half_life, 1.0))
            weights /= weights.sum()
            mean = float(np.sum(weights * rets))
            raw_sigma = float(np.sqrt(np.sum(weights * (rets - mean) ** 2)))
        elif vol_estimator == "har" and len(values) >= 7 * 24 * 60:
            medians = {
                "BTC": 1.4311438917727223e-6,
                "ETH": 2.867685981608497e-6,
                "SOL": 3.8338378109778345e-6,
                "XRP": 2.6632866597914587e-6,
                "HYPE": 7.88846415362757e-6,
                "XAU": 5.867489538679576e-7,
                "NVDAX": 1.4991501550638683e-6,
                "GOOGLX": 1.164081734610725e-6,
                "TSLAX": 1.7347543643816527e-6,
                "AAPLX": 1.0505174940150233e-6,
                "WTIOIL": 3.650564515628981e-6,
                "SPCX": 5.905192220115834e-6,
                "SP500": 1.4967615044700127e-7,
            }
            daily_rv = []
            for day in range(7):
                segment = values[day * 1440 : (day + 1) * 1440]
                day_rets = np.log(segment[native_minutes:]) - np.log(
                    segment[:-native_minutes]
                )
                daily_rv.append(float(np.nanmean(day_rets**2)))
            baseline = medians.get(asset, float(np.nanmedian(daily_rv)))
            rv1 = daily_rv[-1]
            rv3 = float(np.nanmean(daily_rv[-3:]))
            rv7 = float(np.nanmean(daily_rv))
            features = np.log(np.maximum([rv1, rv3, rv7], 1e-16) / baseline)
            log_ratio = -0.15244330935347022 + float(
                np.dot([0.46093968441784466, -0.4304299904665286, 0.6867424882178182], features)
            )
            raw_sigma = float(np.sqrt(baseline * np.exp(log_ratio)))
        else:
            raw_sigma = float(np.std(rets, ddof=1))
        per_step_sigma = raw_sigma * sigma_scale
    else:
        per_step_sigma = 1e-3
    # Stable randomness makes close candidates comparable and avoids accidental selection on
    # Monte Carlo noise while retaining an ordinary ensemble for every prompt.
    seed_material = f"{asset}|{start_time}|{time_increment}|{time_length}".encode()
    seed = int.from_bytes(hashlib.blake2b(seed_material, digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    if innovation == "bootstrap" and rets.size > 20:
        standardized = (rets - np.mean(rets)) / max(float(np.std(rets, ddof=1)), 1e-12)
        shocks = rng.choice(standardized, size=(num_simulations, steps), replace=True)
        shocks *= per_step_sigma
    else:
        # Antithetic pairs materially reduce finite-ensemble asymmetry at essentially zero
        # cost. Each innovation column has an exactly balanced empirical distribution.
        half = (num_simulations + 1) // 2
        if innovation == "student5":
            base = rng.standard_t(5, size=(half, steps)) * (3.0 / 5.0) ** 0.5
            base *= per_step_sigma
        else:
            base = rng.normal(0.0, per_step_sigma, size=(half, steps))
        shocks = np.concatenate((base, -base), axis=0)[:num_simulations]
    shocks[:, 0] = 0.0
    if ar1 != 0.0:
        innovation_scale = max(0.0, 1.0 - ar1 * ar1) ** 0.5
        for j in range(2, steps):
            shocks[:, j] = ar1 * shocks[:, j - 1] + innovation_scale * shocks[:, j]
    if drift_lookback > native_minutes and drift_coef != 0.0:
        drift_values = np.asarray(raw.iloc[-drift_lookback:], dtype=float)
        drift_rets = np.log(drift_values[native_minutes:]) - np.log(
            drift_values[:-native_minutes]
        )
        drift = float(np.nanmean(drift_rets)) * drift_coef
        if np.isfinite(drift):
            shocks[:, 1:] += drift
    if martingale:
        shocks[:, 1:] -= 0.5 * per_step_sigma * per_step_sigma
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *[p.tolist() for p in paths])
