"""Seasonal EWMA-volatility price simulator for Synth SN50 (CRYPTO_24H, COM_EQU_24H, CRYPTO_1H).

Per-minute variance = weekly profile (NY local time, fitted offline, agent/artifacts) x deseasonalised
level. The level is forecast from two gap-aware EWMAs of the context's squared returns, the short one
decaying into the long one over the horizon. Within each 15-min slot a minute-of-hour shape; scheduled
macro releases (calendar at $SYNTH_BENCHMARK_CALENDAR) scale the variance around the release; the
weekend variance of non-crypto assets follows the last weekend seen in the context. Paths: Student-t
increments, an atom at zero for unchanged prices, a per-path volatility multiplier (vol-of-vol),
martingale in price. Every default below was chosen on backtests (agent/journal.md).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
SEASON_FILE = os.environ.get("SEASON_FILE", "season.json")


def P(name, default):
    return float(os.environ.get(name, default))


@lru_cache(maxsize=4)
def _profiles(fname):
    d = json.loads((ART / fname).read_text())
    prof = {s: {k: np.asarray(v) for k, v in d[f"profiles_{s}"].items()} for s in (1, 5)}
    zero = {s: {k: np.asarray(v) for k, v in d.get(f"zero_{s}", {}).items()} for s in (1, 5)}
    moh = {k: np.asarray(v) for k, v in d.get("moh", {}).items()}
    return d["nb"], prof, zero, moh


def _week_minute_profile(asset, idx_utc, step, kind=1):
    """Profile value for each UTC timestamp in idx_utc (DatetimeIndex), fitted on `step`-minute returns.
    kind=1: variance multiplier; kind=2: probability that a step return is exactly zero."""
    nb, prof, zero, moh = _profiles(SEASON_FILE)
    p = (prof if kind == 1 else zero)[step].get(asset)
    if p is None and kind == 2:
        return np.zeros(len(idx_utc))
    if p is None:
        return np.ones(len(idx_utc))
    ny = idx_utc.tz_convert("America/New_York")
    b = ny.dayofweek * nb + (ny.hour * 60 + ny.minute) // (1440 // nb)
    out = p[np.asarray(b)]
    q = moh.get(asset)
    if kind == 1 and q is not None and P("MOH", 1.0) > 0:  # within-slot minute-of-hour shape
        out = out * q[np.asarray(idx_utc.minute)] ** P("MOH", 1.0)
    return out


CRYPTO = ("BTC", "ETH", "SOL", "XRP", "HYPE")

# Variance multipliers around scheduled releases, relative to a non-event day at the same clock time.
# Windows in minutes from the release; values = median over 13 assets x past releases of the squared
# L1 ratio (agent/analysis/events.py). Same-minute releases: the max applies.
# Served table: releases before 2026-07-01 only, validated out of sample on the Sep releases, where
# raising it to a power (EV_POW: 1.75 at 1-min steps, 1.5 at 5-min) scored best -- a median ratio understates
# the mean variance of a release minute. EV_SET=all: same fit on all releases < 2026-10-11 (reference).
EV_WIN = np.array([-30, 0, 2, 5, 15, 30, 60, 120, 240])
EV_MULT_ALL = {
    "CPI": [0.85, 26.0, 5.5, 3.0, 1.8, 1.05, 1.1, 1.0],
    "NFP": [0.8, 10.0, 3.4, 2.3, 1.7, 1.1, 1.0, 1.0],
    "FOMC": [1.0, 10.0, 4.8, 5.0, 2.35, 3.8, 2.6, 1.3],
    "GDP": [0.7, 4.8, 1.3, 1.4, 1.15, 1.0, 1.1, 1.0],
    "PCE": [0.7, 4.7, 1.35, 1.45, 1.15, 1.0, 1.1, 1.0],
    "PPI": [0.9, 4.2, 2.2, 2.0, 1.36, 1.0, 1.0, 1.0],
}

EV_MULT = {
    "CPI": [0.8, 13.6, 5.4, 2.9, 1.27, 1.0, 1.1, 1.0],
    "NFP": [0.7, 7.0, 3.1, 2.1, 1.86, 1.1, 1.0, 1.0],
    "FOMC": [1.0, 8.6, 4.2, 3.6, 2.17, 3.6, 2.2, 1.3],
    "GDP": [0.7, 4.8, 1.3, 1.4, 1.15, 1.0, 1.1, 1.0],
    "PCE": [0.7, 4.7, 1.35, 1.45, 1.15, 1.0, 1.1, 1.0],
    "PPI": [1.0, 4.1, 2.3, 2.3, 1.38, 1.0, 1.1, 1.0],
}


@lru_cache(maxsize=2)
def _calendar(path, mtime):
    try:
        ev = pd.read_parquet(path)
        ev = ev[ev["event"].isin(list(EV_MULT))].copy()
        ev["t"] = pd.to_datetime(ev["time_utc"], utc=True)
        return ev[["t", "event", "scope"]].dropna().reset_index(drop=True)
    except Exception:
        return None


def _event_mult(asset, idx_utc, step=5):
    """Per-minute variance multiplier from the scheduled-event calendar (1 where nothing is scheduled)."""
    out = np.ones(len(idx_utc))
    path = os.environ.get("SYNTH_BENCHMARK_CALENDAR")
    if P("EVENTS", 1.0) <= 0 or not path or not os.path.exists(path) or len(idx_utc) == 0:
        return out
    ev = _calendar(path, os.path.getmtime(path))
    if ev is None or ev.empty:
        return out
    lo, hi = idx_utc[0] - pd.Timedelta(minutes=240), idx_utc[-1] + pd.Timedelta(minutes=30)
    ev = ev[(ev["t"] > lo) & (ev["t"] < hi)]
    for t, name, scope in zip(ev["t"], ev["event"], ev["scope"]):
        if scope != "all" and asset not in str(scope).split(","):
            continue
        dt = np.asarray((idx_utc - t).total_seconds() / 60.0)
        k = np.searchsorted(EV_WIN, dt, side="right") - 1
        inwin = (k >= 0) & (k < len(EV_WIN) - 1)
        m = np.ones(len(idx_utc))
        table = EV_MULT_ALL if os.environ.get("EV_SET") == "all" else EV_MULT
        m[inwin] = np.asarray(table[name])[k[inwin]] ** P("EV_POW", 1.75 if step == 1 else 1.5)
        # same-minute releases: max of the bumps; the pre-release lull only where nothing else applies
        out = np.where((m > 1) | (out > 1), np.maximum(out, m), np.minimum(out, m))
    return out


def _weekend(idx_utc):
    """Fri 17:00 -> Sun 18:00 New York: the underlying cash/futures markets are shut."""
    ny = idx_utc.tz_convert("America/New_York")
    d, h = ny.dayofweek, ny.hour
    return np.asarray((d == 5) | ((d == 6) & (h < 18)) | ((d == 4) & (h >= 17)))


def _ewma_level(r2, s, age, hl):
    w = np.exp(-age * np.log(2) / hl)
    return float(np.sum(w * r2) / max(np.sum(w * s), 1e-12))


def forecast(asset, start_time, time_increment, time_length, context_prices):
    """Per-step log-return std over the horizon, and the last valid context price."""
    n_steps = time_length // time_increment
    start = pd.Timestamp(start_time)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    ctx = context_prices
    close = ctx["close"] if isinstance(ctx, pd.DataFrame) else ctx
    close = close.astype(float)
    idx = pd.DatetimeIndex(close.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    vals = close.to_numpy()
    ok = np.isfinite(vals) & (vals > 0)
    last = float(vals[ok][-1])

    # overlapping `step`-minute returns of the last traded price (estimation only; no trade = no move),
    # seasonal weight = profile summed over each return's minutes
    step = 1 if time_increment < 300 else 5
    pm = _week_minute_profile(asset, idx, step) * _event_mult(asset, idx, step)
    lp = np.log(pd.Series(np.where(ok, vals, np.nan)).ffill().to_numpy())
    r = lp[step:] - lp[:-step]
    cum = np.concatenate([[0.0], np.cumsum(pm)])
    s_all = cum[step + 1:] - cum[1:-step]
    good = np.isfinite(r)
    r2, s = r[good] ** 2, s_all[good]
    age = (len(vals) - 1 - np.arange(step, len(vals)))[good].astype(float)
    if r2.size < 30:
        v_s = v_l = 1e-8
    else:
        cap = P("WINS", 30.0) * np.median(r2 / np.maximum(s, 1e-9))
        r2c = np.minimum(r2, cap * s) if cap > 0 else r2
        v_s = _ewma_level(r2c, s, age, P("HL_S", 30 if step == 1 else 120))
        v_l = _ewma_level(r2c, s, age, P("HL_L", 2880))

    # future per-minute variance: profile x level decaying from short to long EWMA
    lead = (start - idx[-1]).total_seconds() / 60.0
    fut = start + pd.to_timedelta(np.arange(time_length // 60) * 60, unit="s")
    k = lead + np.arange(time_length // 60)
    lvl = v_l + (v_s - v_l) * np.exp(-k / P("TAU_MR", 60 if step == 1 else 240))
    var_min = _week_minute_profile(asset, fut, step) * _event_mult(asset, fut, step) * lvl * P("SCALE", 1.0) ** 2
    we_k = P("WE_SCALE", 1.0) ** 2
    gam = P("WE_GAMMA", 1.0)
    if gam > 0 and r2.size > 100:
        # last weekend vs the weekdays in the context, deseasonalised (L1): how far the profile is off now
        wk_ctx = _weekend(idx[step:])[good]
        a_we, a_wd = np.sqrt(r2[wk_ctx]).sum(), np.sqrt(r2[~wk_ctx]).sum()
        b_we, b_wd = np.sqrt(s[wk_ctx]).sum(), np.sqrt(s[~wk_ctx]).sum()
        if a_we > 0 and a_wd > 0 and b_we > 0 and b_wd > 0:
            rho = ((a_we / b_we) / (a_wd / b_wd)) ** 2
            we_k *= float(np.clip(rho, 0.05, 5.0)) ** gam
    if we_k != 1.0 and (asset not in CRYPTO or P("WE_CRYPTO", 0.0) > 0):
        var_min = np.where(_weekend(fut), var_min * we_k, var_min)
    mps = time_increment // 60
    var_step = var_min[: n_steps * mps].reshape(n_steps, mps).sum(axis=1)
    # probability of an unchanged price over each step, at the step's end
    p0 = _week_minute_profile(asset, fut[mps - 1 :: mps][:n_steps], step, kind=2) * P("ZK", 1.0)
    return np.sqrt(var_step), last, np.clip(p0, 0.0, 0.9)


def simulate(asset, start_time, time_increment=300, time_length=86_400, num_simulations=1000, context_prices=None):
    seed = int(os.environ.get("SYNTH_BENCHMARK_SEED", "12345"))
    rng = np.random.default_rng(seed)
    sig, last, p0 = forecast(asset, start_time, time_increment, time_length, context_prices)
    n_steps = sig.size
    nu = P("NU", 5.0)
    eta = P("ETA", 0.2)
    z = rng.standard_t(nu, size=(num_simulations, n_steps)) * np.sqrt((nu - 2) / nu)
    m = np.exp(eta * rng.standard_normal((num_simulations, 1)) - eta**2)
    if p0.max() > 0:  # atom at zero; the moving steps carry the whole step variance
        moving = rng.random((num_simulations, n_steps)) >= p0[None, :]
        z = np.where(moving, z / np.sqrt(1.0 - p0[None, :]), 0.0)
    inc = z * sig[None, :] * m
    inc -= 0.5 * (sig[None, :] * m) ** 2
    logp = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(inc, axis=1)], axis=1)
    paths = last * np.exp(logp)
    return (start_time, time_increment, *paths)
