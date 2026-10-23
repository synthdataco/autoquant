"""v1: zero-drift stochastic-vol fan.

Per prompt, from the 7-day minute context only:
 - deseasonalise 1-min log returns by a per-asset (weekday x hour) vol profile
   fitted offline on the snapshot (artifacts.json, embedded in the repo);
 - level = blend of fast (6h half-life) and slow (3d half-life) EWMA variance
   of the deseasonalised returns (outlier-capped);
 - forward: AR(1) log-vol mean-reverting to the slow level, Student-t shocks,
   re-seasonalised minute by minute, simulated at 1-min then sub-sampled to
   the scoring step. Handles 24h/5min and 1h/1min (and any increment >= 60s).
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

_ART_FILE = os.environ.get("SYNTH_MODEL_ART", "artifacts.json")
_ART = json.loads((Path(__file__).parent / _ART_FILE).read_text())
_EV = os.environ.get("SYNTH_MODEL_EVENT", "1")
NU = float(os.environ.get("SYNTH_MODEL_NU", "5"))
VOL_SD = float(os.environ.get("SYNTH_MODEL_VOLSD", "0.35"))   # stationary sd of log-sigma
FAST_W = float(os.environ.get("SYNTH_MODEL_FASTW", "0.85"))    # weight on fast EWMA at start

# Event vol multipliers (median post-event vol ratio, pooled deseasonalised data; LOEO-checked
# informally by median vs mean agreement). Type -> (sigma multiplier at event minute, decay HL min)
EVENTS = {"CPI": (1.45, 60), "FOMC": (1.8, 60), "NFP": (1.2, 60), "GDP": (1.15, 60),
          "PCE": (1.15, 60), "PPI": (1.15, 60)}
EQUITY_ONLY = {"NYSE_CLOSED"}  # handled via profile already; no forward multiplier


def _event_mult(asset: str, fut_index: pd.DatetimeIndex) -> np.ndarray:
    if _EV != "1":
        return np.ones(len(fut_index))
    path = os.environ.get("SYNTH_BENCHMARK_CALENDAR")
    mult = np.ones(len(fut_index))
    if not path:
        return mult
    try:
        cal = pd.read_parquet(path)
    except Exception:
        return mult
    cal = cal[cal["event"].isin(EVENTS)]
    if cal.empty:
        return mult
    t = pd.to_datetime(cal["time_utc"], utc=True)
    lo, hi = fut_index[0], fut_index[-1]
    keep = (t >= lo) & (t <= hi + pd.Timedelta(minutes=180))
    for et, ev in zip(t[keep], cal["event"][keep]):
        amp, hl = EVENTS[str(ev)]
        mins = (fut_index - et).total_seconds().to_numpy() / 60.0
        after = mins >= -15  # vol builds just before the release print
        add = np.where(after, (amp - 1.0) * 0.5 ** (np.maximum(mins, 0) / hl), 0.0)
        mult = mult * (1.0 + add)
    return mult
USE_SEAS = os.environ.get("SYNTH_MODEL_SEAS", "1") == "1"
USE_SV = os.environ.get("SYNTH_MODEL_SV", "1") == "1"
WIDTH = float(os.environ.get("SYNTH_MODEL_WIDTH", "0"))  # >0 overrides both horizons
_WIDTH_BY_TL = {86400: 0.90, 3600: 0.95}
_PHI_MIN = 0.5 ** (1.0 / float(os.environ.get("SYNTH_MODEL_PHIHL", "2880")))  # vol half-life, min


def _profile(asset: str) -> np.ndarray:
    a = _ART.get(asset)
    if not a:
        return np.ones((7, 24))
    if "profile168" in a:  # hour-of-week resolution
        return np.asarray(a["profile168"], dtype=float).reshape(7, 24)
    return np.asarray(a["profile"], dtype=float)


def _anchor(asset: str) -> str | None:
    a = _ART.get(asset)
    return a.get("anchor") if a else None


def _seas_at(asset: str, prof: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    """Profile lookup honouring the fit anchor (NY for US-session assets)."""
    if _anchor(asset) == "NY":
        try:
            loc = idx.tz_convert("America/New_York")
            return prof[loc.weekday, loc.hour]
        except Exception:
            pass
    return prof[idx.weekday, idx.hour]


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed_env = os.environ.get("SYNTH_BENCHMARK_SEED")
    base = int(seed_env) if seed_env else 987_654_321
    stamp = str(start_time)
    rng = np.random.default_rng(
        np.random.SeedSequence([base & 0xFFFFFFFF, zlib.crc32(f"{asset}|{stamp}".encode())]))

    closes = context_prices["close"]
    last = float(closes.dropna().iloc[-1])
    steps = time_length // time_increment + 1
    prof = _profile(asset) if USE_SEAS else np.ones((7, 24))

    p = np.asarray(closes, dtype=float)
    ok = np.isfinite(p) & (p > 0)
    # A return spanning k untraded minutes is a k-minute move: scale to per-minute.
    prev_ok = np.maximum.accumulate(np.where(ok, np.arange(len(p)), 0))
    dt_min = np.clip(np.arange(len(p)) - prev_ok, 1.0, 240.0)  # dt at each index
    lp = np.where(ok, np.log(np.where(ok, p, 1.0)), np.nan)
    r_raw = np.diff(lp)
    fin = np.isfinite(r_raw)
    r = (r_raw / np.sqrt(dt_min[1:]))[fin]
    # variance contribution of a scaled k-min return is var/k per minute: weight 1/k
    gap_w = 1.0 / dt_min[1:][fin]
    if r.size < 600:  # degenerate context: flat robust vol
        sig = float(np.nanstd(r)) if r.size > 10 else 1e-3
        shocks = rng.normal(0, sig * np.sqrt(time_increment / 60.0),
                             size=(num_simulations, steps))
        shocks[:, 0] = 0.0
        paths = last * np.exp(np.cumsum(shocks, axis=1))
        return (start_time, time_increment, *[row.tolist() for row in paths])

    # seasonal factor of each return (aligned to its end-minute where possible)
    try:
        idx = closes.index[1:][np.isfinite(np.diff(lp))]
        seas_r = _seas_at(asset, prof, idx)
    except Exception:
        seas_r = np.ones_like(r)
    u = r / seas_r
    cap = np.quantile(np.abs(u), 0.999)
    u_c = np.clip(u, -cap, cap)
    sq = pd.Series(u_c * u_c * gap_w, index=getattr(idx, "values", None))
    fast = float(sq.ewm(halflife=float(os.environ.get("SYNTH_MODEL_FASTHL", "360")), adjust=False).mean().iloc[-1])
    slow = float(sq.ewm(halflife=4320, adjust=False).mean().iloc[-1])
    slow = max(slow, 1e-12); fast = max(fast, 1e-12)

    minutes = time_length // 60
    t0 = pd.Timestamp(start_time)
    fut = pd.date_range(t0, periods=minutes, freq="1min")
    seas_f = _seas_at(asset, prof, fut) * _event_mult(asset, fut)

    _w = WIDTH if WIDTH > 0 else _WIDTH_BY_TL.get(time_length, 0.95)
    mu = 0.5 * np.log(slow) + np.log(_w)
    start_log = mu + FAST_W * (0.5 * np.log(fast) + np.log(_w) - mu) if USE_SV else mu
    eps_sd = VOL_SD * np.sqrt(1 - _PHI_MIN ** 2) if USE_SV else 0.0
    log_sig = np.full(num_simulations, start_log)
    rets = np.empty((num_simulations, minutes))
    scale_t = np.sqrt(NU / (NU - 2.0))
    for j in range(minutes):
        z = rng.standard_t(NU, size=num_simulations) / scale_t
        rets[:, j] = np.exp(log_sig) * seas_f[j] * z
        if USE_SV:
            log_sig = mu + _PHI_MIN * (log_sig - mu) + eps_sd * rng.standard_normal(num_simulations)
    cum = np.cumsum(rets, axis=1)
    every = max(1, time_increment // 60)
    sampled = cum[:, every - 1::every][:, : steps - 1]
    log_paths = np.concatenate([np.zeros((num_simulations, 1)), sampled], axis=1)
    paths = last * np.exp(log_paths)
    return (start_time, time_increment, *[row.tolist() for row in paths])
