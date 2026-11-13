"""Shared numerics of the champion: clocks, calendar, context features, variance path, path simulator.

numpy/pandas only. `modeling.py` serves from it; the research scripts build training sets with the
very same functions, so what is fitted is what is served.

Model (per asset and prompt format), for step h of a prompt starting at t0:

    v_h   = sum_{minutes m in step h} P[minute-of-week(m)] * E(m) * exp(kappa * xs[session(m)])   (seasonal)
            * exp( sum_k W_k(h) x_k + b0 + b1 phi(h) )                                             (level)
    W(h)  = softmax_k(alpha_k + beta_k phi(h)),   phi(h) = log1p(minutes ahead / 30)
    x_k   = log of deseasonalised realised-variance estimators of the context (EWMAs, several half-lives)

    paths: r_{p,h} = sqrt(v_h) * exp(g_{p,h} - Var g_h) * z_{p,h},  z ~ N(0,1)
           g_{p,h} = A(h) xi_p + c OU_{p,h} + lam w_{p,h}      (log-vol: permanent + persistent + white)
    plus an MA(1) term theta on the shocks. Every parameter is fitted on CRPS (research/train.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARAMS_DIR = HERE / "params"
NY = "America/New_York"
CALENDAR_ENV = "SYNTH_BENCHMARK_CALENDAR"

EQUITY = ("NVDAX", "TSLAX", "GOOGLX", "AAPLX", "SPCX", "SP500")
COMMODITY = ("XAU", "WTIOIL")
CRYPTO = ("BTC", "ETH", "SOL", "XRP", "HYPE")

# EWMA half-lives of the level features, in minutes.
HALF_LIVES = np.array([2.0, 5.0, 15.0, 45.0, 120.0, 360.0, 1080.0, 3240.0, 10080.0])
N_HL = len(HALF_LIVES)
# Global features: [sq x N_HL, abs x N_HL, parkinson x N_HL, context mean]; then 2 features of the
# TARGET step's own session (EWMAs over that session's minutes only, half-lives in session minutes).
N_GLOB = 3 * N_HL + 1
SESSION_HALF_LIVES = np.array([400.0, 2000.0])
N_FEAT = N_GLOB + len(SESSION_HALF_LIVES)
# Global features that a session change discounts: the three families' half-lives up to 6 h.
FAST_MASK = np.zeros(N_FEAT)
for _fam in range(3):
    FAST_MASK[_fam * N_HL : _fam * N_HL + 6] = 1.0
N_EXTRA = 4
N_SESS = 4
# A fast estimator can be exactly zero on a thin market; it is floored at this share of the context mean.
REL_FLOOR = 0.02
# Absolute floor of a per-minute variance (log-return units): (0.02 bps)^2.
ABS_FLOOR = (0.02e-4) ** 2
MIN_SESSION_MINUTES = 300
PHI_SCALE = 30.0
# A minute's deseasonalised squared return counts for at most this multiple of the context mean.
U_CAP = 400.0
# Below this many minute returns a context is read through its gaps (see context_arrays).
SPARSE_RETURNS = 500
MAX_GAP_MINUTES = 120
# No minute of the week carries less than this share of the average variance.
PROFILE_FLOOR = 0.01


# ----------------------------------------------------------------------------------------------
# Clocks
# ----------------------------------------------------------------------------------------------


def as_utc_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def ny_wall_minutes(index_utc: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since 1970-01-01 on the New York wall clock (unit-agnostic)."""
    naive = index_utc.tz_convert(NY).tz_localize(None)
    return naive.to_numpy().astype("datetime64[m]").astype(np.int64)


def utc_minutes(index_utc: pd.DatetimeIndex) -> np.ndarray:
    return index_utc.tz_localize(None).to_numpy().astype("datetime64[m]").astype(np.int64)


def minute_of_week(wall: np.ndarray, closed_days: np.ndarray | None = None) -> np.ndarray:
    """Monday 00:00 = 0 ... Sunday 23:59 = 10079, on the wall clock given.

    A day in `closed_days` (wall-clock day numbers: an exchange holiday) reads as a Saturday."""
    day = wall // 1440
    dow = (day + 3) % 7  # 1970-01-01 was a Thursday
    if closed_days is not None and len(closed_days):
        dow = np.where(np.isin(day, closed_days), 5, dow)
    return dow * 1440 + wall % 1440


def session_table(asset: str) -> np.ndarray:
    """Session type of each minute of the week (NY clock): 0 core, 1 shoulder, 2 night, 3 weekend."""
    m = np.arange(10080)
    dow, mod = m // 1440, m % 1440
    out = np.full(10080, 2, dtype=np.int64)
    weekday = dow < 5
    if asset in EQUITY:
        out[weekday & (mod >= 570) & (mod < 960)] = 0  # 09:30-16:00
        out[weekday & (((mod >= 240) & (mod < 570)) | ((mod >= 960) & (mod < 1200)))] = 1  # 04:00-09:30, 16:00-20:00
        weekend = ((dow == 4) & (mod >= 1200)) | (dow == 5) | ((dow == 6) & (mod < 1080))
        out[weekend] = 3
    elif asset in COMMODITY:
        out[weekday & (mod >= 480) & (mod < 1020)] = 0  # 08:00-17:00
        out[weekday & ((mod < 480) | (mod >= 1020))] = 1
        out[(dow == 6) & (mod >= 1080)] = 1
        weekend = ((dow == 4) & (mod >= 1020)) | (dow == 5) | ((dow == 6) & (mod < 1080))
        out[weekend] = 3
    else:
        out[weekday & (mod >= 480) & (mod < 1020)] = 0
        out[weekday & ((mod < 480) | (mod >= 1020))] = 1
        out[dow >= 5] = 3
    return out


# ----------------------------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------------------------

_CALENDAR_CACHE: dict = {"key": None, "frame": None}


def load_calendar() -> pd.DataFrame | None:
    """The scheduled-events view at $SYNTH_BENCHMARK_CALENDAR, or None when it cannot be read."""
    path = os.environ.get(CALENDAR_ENV)
    if not path:
        return None
    try:
        key = (path, os.stat(path).st_mtime_ns)
        if _CALENDAR_CACHE["key"] != key:
            frame = pd.read_parquet(path)
            frame = frame[["time_utc", "event", "scope"]].copy()
            frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
            frame["minute"] = utc_minutes(pd.DatetimeIndex(frame["time_utc"]))
            _CALENDAR_CACHE["key"] = key
            _CALENDAR_CACHE["frame"] = frame
        return _CALENDAR_CACHE["frame"]
    except Exception:
        return None


def asset_events(asset: str, calendar: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """event type -> sorted UTC minute stamps of the events that concern `asset`."""
    if calendar is None or len(calendar) == 0:
        return {}
    try:
        scope = calendar["scope"].astype(str)
        mine = (scope == "all") | scope.map(lambda s: asset in s.split(","))
        sub = calendar[mine]
        return {e: np.sort(g["minute"].to_numpy()) for e, g in sub.groupby("event")}
    except Exception:
        return {}


def closed_days_of(events: dict[str, np.ndarray]) -> np.ndarray:
    """NY wall-clock day numbers on which the asset's exchange is closed (NYSE_CLOSED rows)."""
    stamps = events.get("NYSE_CLOSED")
    if stamps is None or len(stamps) == 0:
        return np.empty(0, dtype=np.int64)
    idx = pd.DatetimeIndex(stamps.astype("datetime64[m]")).tz_localize("UTC")
    return np.unique(ny_wall_minutes(idx) // 1440)


def event_multiplier(utc_min: np.ndarray, events: dict[str, np.ndarray], table: dict | None) -> np.ndarray:
    """Variance multiplier of each minute from the scheduled events (1 where none).

    `table[event] = {"edges": [e0, e1, ...], "mult": [m0, m1, ...]}`: minutes since the event in
    [e_i, e_{i+1}) carry m_i. Where several events overlap the largest multiplier applies (macro
    releases share a minute; multiplying them would double count)."""
    out = np.ones(len(utc_min))
    if not table or not events or len(utc_min) == 0:
        return out
    for name, spec in table.items():
        stamps = events.get(name)
        if stamps is None or len(stamps) == 0:
            continue
        edges = np.asarray(spec["edges"], dtype=np.int64)
        mult = np.asarray(spec["mult"], dtype=float)
        pos = np.searchsorted(stamps, utc_min, side="right") - 1  # last event at or before the minute
        has = pos >= 0
        since = np.where(has, utc_min - stamps[np.maximum(pos, 0)], -1)
        bucket = np.searchsorted(edges, since, side="right") - 1
        ok = has & (since >= edges[0]) & (bucket < len(mult))
        out[ok] = np.maximum(out[ok], mult[bucket[ok]])
    return out


# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------

_PARAM_CACHE: dict = {}


def load_params(asset: str, time_length: int, params_dir: Path | None = None) -> dict:
    """Profile + fitted parameters of (asset, format). Falls back to a generic set for an unknown asset."""
    # CORE_PARAMS_DIR: research override (candidate parameter sets); the champion serves PARAMS_DIR.
    root = Path(params_dir) if params_dir is not None else Path(os.environ.get("CORE_PARAMS_DIR") or PARAMS_DIR)
    tag = "1h" if time_length <= 3600 else "24h"
    key = (str(root), asset, tag)
    if key not in _PARAM_CACHE:
        prof_path = root / f"profile_{asset}.json"
        par_path = root / f"model_{asset}_{tag}.json"
        prof = json.loads(prof_path.read_text()) if prof_path.exists() else {}
        par = json.loads(par_path.read_text()) if par_path.exists() else {}
        out = default_params()
        out.update(par)
        out["profile"] = np.maximum(np.asarray(prof.get("profile", np.ones(10080)), dtype=float), PROFILE_FLOOR)
        out["utc_profile"] = np.asarray(prof.get("utc_profile", np.ones((2, 1440))), dtype=float).reshape(2, 1440)
        out["events"] = prof.get("events", {})
        out["sessions"] = session_table(asset)
        for name in ("alpha", "beta", "fam_bias", "ce0", "ce1"):
            out[name] = np.asarray(out[name], dtype=float)
        _PARAM_CACHE[key] = out
    return _PARAM_CACHE[key]


def default_params() -> dict:
    """An untrained but sane parameter set: geometric mean of mid-horizon EWMAs, mild fat tails."""
    alpha = np.full(N_FEAT, -4.0)
    alpha[3:7] = 0.0
    return {
        "alpha": alpha.tolist(),
        "beta": np.zeros(N_FEAT).tolist(),
        "fam_bias": [0.0, 0.0, 0.0],
        "b0": 0.0,
        "b1": 0.0,
        "kappa": 0.0,
        "A0": 0.15,
        "A1": 0.05,
        "c": 0.2,
        "rho": 0.9,
        "lam": 0.3,
        "theta": 0.0,
        "ev_pow": 1.0,
        "p": 0.0,
        "gamma": 0.0,
        "ce0": [0.0] * N_EXTRA,
        "ce1": [0.0] * N_EXTRA,
    }


# ----------------------------------------------------------------------------------------------
# Context features
# ----------------------------------------------------------------------------------------------


def dense_minutes(index_utc: pd.DatetimeIndex, columns: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Put a context on a gap-free minute grid (missing minutes -> NaN). Returns (utc minutes, columns)."""
    minutes = utc_minutes(index_utc)
    if len(minutes) == 0:
        return minutes, columns
    span = int(minutes[-1] - minutes[0]) + 1
    if span == len(minutes) and (len(minutes) < 2 or np.all(np.diff(minutes) == 1)):
        return minutes, columns
    grid = np.arange(minutes[0], minutes[0] + span)
    pos = (minutes - minutes[0]).astype(np.int64)
    out = {}
    for name, values in columns.items():
        dense = np.full(span, np.nan)
        dense[pos] = values
        out[name] = dense
    return grid, out


_W_CACHE: dict = {}


def _weights(n: int) -> np.ndarray:
    """[N_HL, n] EWMA weights, the last column being the newest minute."""
    size = _W_CACHE.get("size", 0)
    if size < n:
        size = max(n, 30 * 1440 + 8)
        age = np.arange(size - 1, -1, -1, dtype=float)
        _W_CACHE["w"] = np.exp2(-age[None, :] / HALF_LIVES[:, None])
        _W_CACHE["size"] = size
    return _W_CACHE["w"][:, _W_CACHE["size"] - n :]


def _session_weights(n: int) -> np.ndarray:
    """[len(SESSION_HALF_LIVES), n] EWMA weights over a session's own minutes, newest last."""
    size = _W_CACHE.get("ssize", 0)
    if size < n:
        size = max(n, 30 * 1440 + 8)
        age = np.arange(size - 1, -1, -1, dtype=float)
        _W_CACHE["sw"] = np.exp2(-age[None, :] / SESSION_HALF_LIVES[:, None])
        _W_CACHE["ssize"] = size
    return _W_CACHE["sw"][:, _W_CACHE["ssize"] - n :]


def features_from_arrays(
    r: np.ndarray, pk: np.ndarray, seas: np.ndarray, sess: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Level features from minute arrays that END at the last bar the prompt may see.

    r     log return of each minute (NaN where either close is missing)
    pk    Parkinson variance of each minute bar, ln(high/low)^2 / (4 ln 2) (NaN where unknown)
    seas  seasonal x event variance multiplier of each minute
    sess  session type of each minute
    Returns (x [N_GLOB] log-variances per minute, xs [N_SESS] log session ratios, info) where info holds
    xz [N_SESS] zero-return shares, xsl [N_SESS, 2] log session levels, cur (session of the last minute),
    xe [N_EXTRA] extras.
    """
    n = len(r)
    ok = np.isfinite(r)
    info = {"n_valid": int(ok.sum()), "xz": np.ones(N_SESS), "mean_u": 0.0, "cur": int(sess[-1]) if n else 0,
            "xsl": np.full((N_SESS, len(SESSION_HALF_LIVES)), np.log(ABS_FLOOR)), "xe": np.zeros(N_EXTRA)}
    if info["n_valid"] < 30:
        x = np.full(N_GLOB, np.log(ABS_FLOOR))
        return x, np.zeros(N_SESS), info
    inv = 1.0 / np.maximum(seas, 1e-12)
    u = np.where(ok, r * r * inv, 0.0)
    a = np.where(ok, np.abs(r) * np.sqrt(inv), 0.0)
    okp = np.isfinite(pk)
    p = np.where(okp, pk * inv, 0.0)
    okf, okpf = ok.astype(float), okp.astype(float)
    cap = U_CAP * u.sum() / okf.sum()            # one print 20 sigmas out must not own the forecast
    if cap > 0:
        u, a, p = np.minimum(u, cap), np.minimum(a, np.sqrt(cap)), np.minimum(p, cap)
    mean_u = u.sum() / okf.sum()
    floor = max(REL_FLOOR * mean_u, ABS_FLOOR)
    w = _weights(n)
    den = w @ okf
    denp = w @ okpf
    safe = den > 1e-9
    sq = np.where(safe, (w @ u) / np.where(safe, den, 1.0), mean_u)
    ab = np.where(safe, ((w @ a) / np.where(safe, den, 1.0)) ** 2 * (np.pi / 2.0), mean_u)
    safep = denp > 1e-9
    pa = np.where(safep, (w @ p) / np.where(safep, denp, 1.0), sq)
    x = np.log(np.concatenate([sq, ab, pa, [mean_u]]) + floor)
    xs = np.zeros(N_SESS)
    zero = ok & (r == 0.0)
    xz = np.full(N_SESS, float(zero.sum()) / float(ok.sum()))
    xsl = np.full((N_SESS, len(SESSION_HALF_LIVES)), np.log(mean_u + floor))
    for c in range(N_SESS):
        sel = ok & (sess == c)
        cnt = int(sel.sum())
        if cnt >= MIN_SESSION_MINUTES:
            uc = u[sel]
            xs[c] = np.log((uc.sum() / cnt + floor) / (mean_u + floor))
            xz[c] = float(zero[sel].sum()) / cnt
            ww = _session_weights(cnt)                          # ages in minutes OF THAT SESSION
            xsl[c] = np.log((ww @ uc) / ww.sum(axis=1) + floor)
    info["xz"], info["xsl"], info["mean_u"] = xz, xsl, float(mean_u)
    xe = np.zeros(N_EXTRA)
    r0 = np.where(ok, r, 0.0)
    s60, q60 = r0[-60:].sum(), (r0[-60:] ** 2).sum()
    s1d, q1d = r0[-1440:].sum(), (r0[-1440:] ** 2).sum()
    xe[0] = np.clip(s60 / np.sqrt(q60), -4.0, 4.0) if q60 > 0 else 0.0
    xe[1] = abs(xe[0])
    xe[2] = np.clip(s1d / np.sqrt(q1d), -4.0, 4.0) if q1d > 0 else 0.0
    u120 = u[-120:]
    xe[3] = np.log((u120.max() + floor) / (u120.mean() + floor))
    info["xe"] = xe
    return x, xs, info


def context_arrays(context: pd.DataFrame, asset: str, params: dict, events: dict[str, np.ndarray]):
    """Minute arrays of a context, cut at its last real close. Returns None when it holds no price."""
    idx = as_utc_index(context.index)
    cols = {"close": pd.to_numeric(context["close"], errors="coerce").to_numpy(dtype=float)}
    for name in ("high", "low"):
        if name in context.columns:
            cols[name] = pd.to_numeric(context[name], errors="coerce").to_numpy(dtype=float)
    minutes, cols = dense_minutes(idx, cols)
    close = cols["close"]
    good = np.isfinite(close) & (close > 0)
    if not good.any():
        return None
    last = int(np.flatnonzero(good)[-1])
    close, minutes = close[: last + 1], minutes[: last + 1]
    logc = np.log(np.where(good[: last + 1], close, np.nan))
    r = np.full(last + 1, np.nan)
    r[1:] = np.diff(logc)
    pk = np.full(last + 1, np.nan)
    if "high" in cols and "low" in cols:
        hi, lo = cols["high"][: last + 1], cols["low"][: last + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            okb = np.isfinite(hi) & np.isfinite(lo) & (lo > 0) & (hi >= lo)
            pk = np.where(okb, np.log(np.where(okb, hi, 1.0) / np.where(okb, lo, 1.0)) ** 2 / (4.0 * np.log(2.0)), np.nan)
    seas, sess = seasonal_of(minutes, asset, params, events)
    if int(np.isfinite(r).sum()) < SPARSE_RETURNS:
        # A context with few adjacent prints (a sparse feed): read the return across each short gap and
        # spread its square over the gap's minutes in proportion to the seasonal. No price is invented;
        # the alternative is a forecast from almost nothing.
        valid = np.flatnonzero(good[: last + 1])
        for i, j in zip(valid[:-1], valid[1:]):
            if 1 < j - i <= MAX_GAP_MINUTES:
                step = logc[j] - logc[i]
                share = seas[i + 1 : j + 1] / seas[i + 1 : j + 1].sum()
                r[i + 1 : j + 1] = np.sign(step) * np.sqrt(step * step * share)
    return {"r": r, "pk": pk, "seas": seas, "sess": sess, "price": float(close[-1]), "last_minute": int(minutes[-1]),
            "tick": tick_size(close)}


def seasonal_of(minutes: np.ndarray, asset: str, params: dict, events: dict[str, np.ndarray]):
    """(seasonal x event variance multiplier, session) of UTC minute stamps: New-York-clock weekly
    profile x UTC-clock daily profile x scheduled events."""
    wall = ny_wall_minutes(pd.DatetimeIndex(minutes.astype("datetime64[m]")).tz_localize("UTC"))
    mow = minute_of_week(wall, closed_days_of(events) if asset in EQUITY else None)
    weekend = ((minutes // 1440 + 3) % 7 >= 5).astype(np.int64)
    seas = params["profile"][mow] * params["utc_profile"][weekend, minutes % 1440]
    seas = seas * event_multiplier(minutes, events, params.get("events")) ** params.get("ev_pow", 1.0)
    return seas, params["sessions"][mow]


def future_seasonal(start_minute: int, n_minutes: int, asset: str, params: dict, events: dict[str, np.ndarray]):
    """Seasonal x event multiplier and session of the minutes start+1 ... start+n (bar labels)."""
    return seasonal_of(start_minute + 1 + np.arange(n_minutes, dtype=np.int64), asset, params, events)


# ----------------------------------------------------------------------------------------------
# Variance path and simulator
# ----------------------------------------------------------------------------------------------


def step_sessions(sess: np.ndarray, inc_minutes: int) -> np.ndarray:
    """Session of each step: that of its last minute."""
    steps = len(sess) // inc_minutes
    return sess[inc_minutes - 1 :: inc_minutes][:steps]


def variance_path(
    x: np.ndarray, xs: np.ndarray, info: dict, seas: np.ndarray, sess: np.ndarray, inc_minutes: int, params: dict
) -> np.ndarray:
    """Variance of each step's log return, [H]."""
    steps = len(seas) // inc_minutes
    xz = info["xz"]
    minute_var = seas * np.exp(params["kappa"] * xs[sess])
    s = minute_var[: steps * inc_minutes].reshape(steps, inc_minutes).sum(axis=1)
    ahead = (np.arange(steps) + 0.5) * inc_minutes
    phi = np.log1p(ahead / PHI_SCALE)
    ss = step_sessions(sess, inc_minutes)
    fam = np.concatenate([np.repeat(params["fam_bias"], N_HL), [0.0]])
    feat = np.concatenate([np.broadcast_to(x + fam, (steps, N_GLOB)), info["xsl"][ss]], axis=1)  # [H, N_FEAT]
    logits = params["alpha"][None, :] + params["beta"][None, :] * phi[:, None]
    cross = (ss != info["cur"]).astype(float)
    logits = logits - params.get("gamma", 0.0) * cross[:, None] * FAST_MASK[None, :]
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)
    p = float(params.get("p", 0.0))
    if abs(p) < 1e-3:
        level = (w * feat).sum(axis=1)
    else:
        z = p * feat
        zmax = z.max(axis=1, keepdims=True)
        level = (zmax[:, 0] + np.log((w * np.exp(z - zmax)).sum(axis=1))) / p      # power mean of the estimators
    level = level + params["b0"] + params["b1"] * phi
    dlt, dlt1 = np.asarray(params.get("dlt", np.zeros(N_SESS))), np.asarray(params.get("dlt1", np.zeros(N_SESS)))
    level = level + dlt[ss] + dlt1[ss] * phi + params.get("gl", 0.0) * xz[ss]
    ce0, ce1 = params.get("ce0"), params.get("ce1")
    if ce0 is not None:
        level = level + float(np.dot(ce0, info["xe"])) + float(np.dot(ce1, info["xe"])) * phi
    return s * np.exp(level)


def softplus(v):
    return np.logaddexp(0.0, v)


def white_logvol(ss: np.ndarray, xz: np.ndarray, params: dict) -> np.ndarray:
    """Sd of the white log-vol noise of each step (session- and liquidity-dependent tail weight)."""
    if "lam_raw" not in params:
        return np.full(len(ss), float(params["lam"]))
    raw = np.asarray(params["lam_raw"], dtype=float)
    return softplus(raw[ss] + params.get("gz", 0.0) * xz[ss])


def logvol_std(steps: int, inc_minutes: int, params: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(A(h), sd of the OU part at h, rho per step)."""
    ahead = (np.arange(steps) + 0.5) * inc_minutes
    a = params["A0"] + params["A1"] * np.log1p(ahead / PHI_SCALE)
    rho = float(params["rho"]) ** (inc_minutes / 5.0)  # rho is quoted per 5 minutes
    h = np.arange(1, steps + 1)
    ou_sd = params["c"] * np.sqrt(1.0 - rho ** (2 * h))
    return a, ou_sd, rho


def ndtri(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation, |rel err| < 1.2e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = np.sqrt(-2.0 * np.log(p[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return out


_STRATA: dict = {}


def normals(n: int, cols: int, rng: np.random.Generator, sampler: str) -> np.ndarray:
    """[n, cols] standard normals. "lhs": every column is a random permutation of the n stratum
    midpoints (each marginal is the best n-point picture of N(0,1)); "iid": plain draws."""
    if sampler == "iid" or n < 8:
        return rng.standard_normal((n, cols))
    if n not in _STRATA:
        base = ndtri((np.arange(n) + 0.5) / n)
        _STRATA[n] = base / np.sqrt(np.mean(base * base))
    return rng.permuted(np.repeat(_STRATA[n][:, None], cols, axis=1), axis=0)


def simulate_log_returns(
    v: np.ndarray, lam: np.ndarray, inc_minutes: int, params: dict, n: int, rng: np.random.Generator, sampler: str = "lhs",
    return_var: bool = False,
):
    """[n, H] log returns. `lam` [H]: sd of the white log-vol noise of each step."""
    steps = len(v)
    a, ou_sd, rho = logvol_std(steps, inc_minutes, params)
    c, theta = float(params["c"]), float(params["theta"])
    xi = normals(n, 1, rng, sampler)
    g = a[None, :] * xi
    if c > 0:
        eta = normals(n, steps, rng, sampler) * np.sqrt(1.0 - rho * rho)
        ou = np.empty((n, steps))
        prev = np.zeros(n)
        for h in range(steps):
            prev = rho * prev + eta[:, h]
            ou[:, h] = prev
        g = g + c * ou
    g = g + lam[None, :] * normals(n, steps, rng, sampler)
    var_g = a * a + ou_sd * ou_sd + lam * lam
    sd = np.sqrt(v)[None, :] * np.exp(g - var_g[None, :])
    if sampler == "anti":
        half = rng.standard_normal(((n + 1) // 2, steps))
        z = np.concatenate([half, -half], axis=0)[:n]
        sd[(n + 1) // 2 :] = sd[: n - (n + 1) // 2]
    else:
        z = normals(n, steps, rng, sampler)
    e = sd * z
    if theta != 0.0:
        e[:, 1:] = (e[:, 1:] + theta * e[:, :-1]) / np.sqrt(1.0 + theta * theta)
    out = e - 0.5 * sd * sd
    return (out, sd * sd) if return_var else out


def quantile_match(r: np.ndarray, ref: np.ndarray, weights: np.ndarray, levels: tuple[int, ...]) -> np.ndarray:
    """Hierarchical quantile matching of a served ensemble against a larger reference sample of the same model.

    For each block size k (fine -> coarse) every block's return across the n served paths is moved, rank by
    rank, onto the reference sample's quantiles at the n stratum medians; a path's correction is spread
    over the block's steps in proportion to `weights` (its own conditional variances). A coarse correction
    moves a fine step by O(1/sqrt(k)) of a fine one, so finer levels stay matched. Removes most of the
    ensemble's sampling noise at every scored interval length without touching the model.
    """
    n, steps = r.shape
    m = ref.shape[0]
    if m < 2 * n:
        return r
    mid = ((np.arange(n) + 0.5) * m / n).astype(np.int64)
    lo = np.clip(mid - 1, 0, m - 1)
    r = r.copy()
    for k in levels:
        nb = steps // k
        if nb == 0:
            continue
        cut = nb * k
        cur = r[:, :cut].reshape(n, nb, k).sum(axis=2)
        big = np.sort(ref[:, :cut].reshape(m, nb, k).sum(axis=2), axis=0)
        target = 0.5 * (big[lo] + big[mid])
        order = np.argsort(cur, axis=0, kind="stable")
        delta = np.empty_like(cur)
        np.put_along_axis(delta, order, target, axis=0)
        delta -= cur
        w = weights[:, :cut].reshape(n, nb, k)
        w = w / np.maximum(w.sum(axis=2, keepdims=True), 1e-300)
        r[:, :cut] += (delta[:, :, None] * w).reshape(n, cut)
    return r


def tick_size(close: np.ndarray, lookback: int = 3000) -> float:
    """Price increment of the venue, read off the context's own closes (0.0 when it cannot be told)."""
    c = close[np.isfinite(close) & (close > 0)][-lookback:]
    if len(c) < 50:
        return 0.0
    for decimals in range(0, 10):
        scaled = c * 10.0**decimals
        if np.all(np.abs(scaled - np.round(scaled)) < 1e-6 * np.maximum(1.0, scaled) + 1e-9):
            steps = np.abs(np.diff(np.round(scaled).astype(np.int64)))
            steps = steps[steps > 0]
            unit = int(np.gcd.reduce(steps)) if len(steps) else 1
            return unit / 10.0**decimals
    return 0.0
