"""Seasonal-EWMA scale-mixture model for Synth SN50 price fans.

Per prompt:
  1. log returns from the context (NaN dropped, never interpolated), deseasonalised by a fixed
     weekly volatility profile keyed on New York local time (minute of week), fitted offline
     (agent/profiles.npz: v5 = 5-min return variance shape, v1 = 1-min shape, mean 1 over a week);
  2. fast and slow EWMA of the deseasonalised variance; the forecast level reverts from the fast
     to the slow one over the horizon;
  3. per-step variance = level x profile at that step's minute of week;
  4. paths: Student-t innovations, times a per-path log-normal variance multiplier (vol uncertainty).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Days of minute history simulate() receives as context_prices: a plain int from 3 to 30.
CONTEXT_DAYS = 14

PARAMS = {
    "hl_fast": 240.0,  # EWMA half-life, minutes
    "hl_slow": 2880.0,
    "t_mr": 360.0,  # minutes for the forecast level to revert from fast to slow
    "w_fast0": 0.6,  # weight of the fast EWMA at lead 0 (rest on the slow one)
    "mix_sd": 0.35,  # sd of the per-path log variance multiplier
    "nu": 5.0,  # Student-t dof of the step innovations
    "clip": 30.0,  # cap on a deseasonalised squared return, in multiples of the slow level
    "prof_floor": 0.03,  # floor on the profile (fraction of its weekly mean)
    "scale": 1.0,  # global variance scale
}

_HERE = Path(__file__).resolve().parent
_PROFILES: dict | None = None
_WEEK = 7 * 1440


def _profiles() -> dict:
    global _PROFILES
    if _PROFILES is None:
        try:
            with np.load(_HERE / "profiles.npz") as z:
                _PROFILES = {k: z[k].astype(np.float64) for k in z.files}
        except OSError:
            _PROFILES = {}
    return _PROFILES


def _profile(asset: str, kind: str) -> np.ndarray:
    p = _profiles().get(f"{asset}_{kind}")
    if p is None:
        return np.ones(_WEEK)
    return np.maximum(p, PARAMS["prof_floor"])


def _minute_of_week_ny(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert("America/New_York")
    return (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()


def _ewma(values: np.ndarray, ages: np.ndarray, half_life: float) -> float:
    w = np.exp2(-ages / half_life)
    s = w.sum()
    return float((w * values).sum() / s) if s > 0 else float("nan")


def _seed() -> int:
    raw = os.environ.get("SYNTH_BENCHMARK_SEED")
    try:
        return int(raw) if raw is not None else 12345
    except ValueError:
        return 12345


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    p = PARAMS
    seed = _seed()
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start_time)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    n_steps = time_length // time_increment
    step_min = max(time_increment // 60, 1)

    close = context_prices["close"] if isinstance(context_prices, pd.DataFrame) else context_prices
    close = pd.to_numeric(close, errors="coerce")
    close = close[np.isfinite(close.to_numpy(dtype=float)) & (close.to_numpy(dtype=float) > 0)]
    last = float(close.iloc[-1])

    kind = "v1" if step_min == 1 else "v5"
    k = 1 if step_min == 1 else 5
    prof = _profile(asset, kind)

    # deseasonalised variance per minute from the context's k-minute returns (overlapping)
    lp = pd.Series(np.log(close.to_numpy(dtype=float)), index=close.index)
    full = lp.reindex(pd.date_range(lp.index[0], lp.index[-1], freq="1min"))
    v = full.to_numpy()
    rk = v[k:] - v[:-k]
    idx = full.index[:-k]
    ok = np.isfinite(rk)
    rk, idx = rk[ok], idx[ok]
    if rk.size >= 30:
        mow = _minute_of_week_ny(idx)
        z = rk**2 / (k * prof[mow])
        ages = (full.index[-1] - idx).total_seconds().to_numpy() / 60.0
        slow = _ewma(z, ages, p["hl_slow"])
        z = np.minimum(z, p["clip"] * slow)
        slow = _ewma(z, ages, p["hl_slow"])
        fast = _ewma(z, ages, p["hl_fast"])
    else:
        slow = fast = (1e-3) ** 2 / 1.0  # ~1bp per minute fallback

    # per-step variance over the horizon
    first = t0 + pd.Timedelta(minutes=0)
    step_starts = pd.date_range(first, periods=n_steps, freq=f"{step_min}min")
    mow_f = _minute_of_week_ny(step_starts)
    lead = (np.arange(n_steps) + 0.5) * step_min
    wf = p["w_fast0"] * np.exp(-lead / p["t_mr"])
    level = wf * fast + (1 - wf) * slow
    if kind == "v5" and step_min != 5:
        seas = np.array([prof[(m + np.arange(step_min)) % _WEEK].mean() for m in mow_f])
    else:
        seas = prof[mow_f]
    var = p["scale"] * level * seas * step_min  # per-step variance (log-return^2)

    # scale mixture: per-path variance multiplier with mean 1
    sd = p["mix_sd"]
    mult = np.exp(sd * rng.standard_normal(num_simulations) - 0.5 * sd * sd)
    nu = p["nu"]
    eps = rng.standard_t(nu, size=(num_simulations, n_steps)) * np.sqrt((nu - 2) / nu)
    sig = np.sqrt(var)[None, :] * np.sqrt(mult)[:, None]
    steps = eps * sig - 0.5 * sig**2
    logp = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(steps, axis=1)], axis=1)
    paths = last * np.exp(logp)
    return (start_time, time_increment, *paths)
