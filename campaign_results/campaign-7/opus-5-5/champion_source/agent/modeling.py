"""Seasonal HAR-EWMA scale-mixture model for Synth SN50 price fans.

Per prompt:
  1. k-minute log returns from the context (k = the prompt's step: 5 for 24h, 1 for 1h; NaN
     dropped, never interpolated), deseasonalised by a fixed weekly volatility profile keyed on
     New York local time (minute of week), fitted offline (agent/profiles.npz);
  2. three EWMAs of the deseasonalised variance (fast / medium / slow half-lives); the level
     forecast at each lead is a lead-dependent convex mix of them (QLIKE-fitted offline);
  3. per-step variance = scale x level(lead) x profile(minute of week of the step);
  4. paths: Student-t innovations times a per-path log-normal variance multiplier.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Days of minute history simulate() receives as context_prices: a plain int from 3 to 30.
CONTEXT_DAYS = 21

CRYPTO = {"BTC", "ETH", "SOL", "XRP", "HYPE"}

# Per (group, horizon) settings. hl: EWMA half-lives in minutes; knots: leads (minutes) at which
# the mixing weights w (one row per knot, over the three EWMAs) are given; linear in between.
GROUPS = {
    "crypto_24h": {
        "hl": [60.0, 360.0, 11520.0],
        "knots": [30.0, 150.0, 480.0, 1080.0],
        "w": [[0.45, 0.40, 0.15], [0.25, 0.55, 0.20], [0.05, 0.70, 0.25], [0.0, 0.70, 0.30]],
    },
    "equity_24h": {
        "hl": [60.0, 360.0, 11520.0],
        "knots": [30.0, 150.0, 480.0, 1080.0],
        "w": [[0.65, 0.25, 0.10], [0.40, 0.35, 0.25], [0.10, 0.55, 0.35], [0.05, 0.45, 0.50]],
    },
    "crypto_1h": {
        "hl": [10.0, 60.0, 480.0],
        "knots": [2.5, 10.0, 22.5, 45.0],
        "w": [[0.55, 0.20, 0.25], [0.40, 0.20, 0.40], [0.30, 0.25, 0.45], [0.20, 0.25, 0.55]],
    },
}

PARAMS = {
    "mix_sd": 0.35,  # sd of the per-path log variance multiplier
    "mix_rw": 0.0,  # sd per sqrt(day) of a random walk in the per-path log variance (lead-growing)
    "mix_state": 0.0,  # extra mixing sd per unit |ln(fast/slow level)| (uncertainty when regimes move)
    "strat": True,  # stratified (Latin-hypercube) draws for the multiplier and the innovations
    "nu": 5.0,  # Student-t dof of the step innovations
    "clip": 30.0,  # cap on a deseasonalised squared return, in multiples of its 2-day EWMA
    "prof_floor": 0.03,  # floor on the profile (fraction of its weekly mean)
    "prof_pow": 1.0,  # profile shape exponent (1 = as fitted; <1 flattens the day)
    "we_mult": 1.0,  # extra variance multiplier on weekend steps (Fri 17:00 -> Sun 18:00 NY)
    "hl_override": None,  # replace the group's three EWMA half-lives (minutes); None = as fitted
    "we_mix_sd": 0.5,  # extra per-path log-variance dispersion on weekend steps (weekends vary ~10x)
    "we_mix_sd_equ": 0.8,  # the same for the equity/commodity perps (their weekends are near-dead)
    "we_wts": None,  # EWMA mix (fast, med, slow) for weekend steps; None = the lead-dependent one
    "scale": 1.0,  # global variance scale
    "est": "abs",  # level estimator: "sq" (mean squared return) or "abs" (mean absolute return)
    "abs_scale": 1.2,  # extra variance scale for "abs" (sigma = sqrt(pi/2) x mean |r| for a Gaussian)
    "prof_kind": None,  # profile shape: "v" (mean squared) or "a" (mean absolute); None = follow est
    "range_w": 0.0,  # weight of the high-low range in the absolute-return level measure (off: no gain)
    "tick_round": True,  # round simulated prices to the price grid seen in the context
    "zero_mult_tick": 0.5,  # zero_mult to use instead when the prices are rounded (the grid makes zeros)
    "zero_mult": 1.0,  # multiplier on the profiled share of exactly-zero returns (0 = continuous)
    "sv_sd": 0.0,  # sd of a fast per-step log-variance AR(1) (intra-path vol bursts); 0 = off
    "sv_hl": 5.0,  # its half-life, in steps
    "we_alpha": 0.0,  # weight of the weekend-only level on weekend steps (Fri 17:00 -> Sun 18:00 NY)
    "we_hl": 2880.0,  # half-life (wall minutes) of the weekend-only EWMA
    "profile_file": "profiles.npz",
    "1h_zero_mult_tick": 0.0,
    "1h_we_mix_sd": 0.0,
    "1h_we_mix_sd_equ": 0.0,
    "1h_mix_sd": 0.6,  # wider per-path vol dispersion for 1h (realised-vol PIT was U-shaped) ...
    "1h_nu": 7.0,  # ... with thinner step tails to keep each step's marginal
    "events_file": "events.json",  # variance multipliers after scheduled releases ("" = off)
    "ev_stat": "med",  # "med" (median-based) or "mean" (trimmed mean) multipliers
    "ev_pow": 1.0,  # shrink: multiplier ** ev_pow
}

CALENDAR_ENV = "SYNTH_BENCHMARK_CALENDAR"

_HERE = Path(__file__).resolve().parent
_PROFILES: dict = {}
_WEEK = 7 * 1440


def _profile(asset: str, kind: str, floor: bool = True) -> np.ndarray | None:
    name = PARAMS["profile_file"]
    if name not in _PROFILES:
        try:
            with np.load(_HERE / name) as z:
                _PROFILES[name] = {k: z[k].astype(np.float64) for k in z.files}
        except OSError:
            _PROFILES[name] = {}
    prof = _PROFILES[name].get(f"{asset}_{kind}")
    if not floor:
        return prof
    if prof is None:
        return np.ones(_WEEK)
    prof = np.maximum(prof, PARAMS["prof_floor"])
    if PARAMS["prof_pow"] != 1.0:
        prof = prof ** PARAMS["prof_pow"]
        prof = prof / prof.mean()
    return prof


_EVENTS: dict = {}
_CALENDAR: dict = {}


def _event_table() -> dict:
    name = PARAMS["events_file"]
    if not name:
        return {}
    if name not in _EVENTS:
        try:
            import json

            _EVENTS[name] = json.loads((_HERE / name).read_text())
        except (OSError, ValueError):
            _EVENTS[name] = {}
    return _EVENTS[name]


def _calendar() -> pd.DataFrame | None:
    """The scheduled-event calendar the harness/miner provides, re-read when the file changes."""
    path = os.environ.get(CALENDAR_ENV)
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _CALENDAR.get("key") != (path, mtime):
        try:
            cal = pd.read_parquet(path, columns=["time_utc", "event", "scope"])
            cal["t"] = pd.to_datetime(cal["time_utc"], utc=True)
            cal = cal.dropna(subset=["t"]).sort_values("t")
        except Exception:  # a broken calendar must never cost a prompt
            cal = None
        _CALENDAR.update(key=(path, mtime), cal=cal)
    return _CALENDAR["cal"]


def _event_multiplier(asset: str, start: pd.Timestamp, n_minutes: int, p: dict) -> np.ndarray | None:
    """Variance multiplier for each 1-min return label in [start, start + n_minutes) (None = all 1)."""
    table = _event_table()
    cal = _calendar() if table else None
    if cal is None or cal.empty:
        return None
    grp = "crypto" if asset in CRYPTO else "equ"
    lo, hi = start - pd.Timedelta(days=1), start + pd.Timedelta(minutes=n_minutes + 1)
    sel = cal[(cal["t"] > lo) & (cal["t"] <= hi)]
    if sel.empty:
        return None
    mult = np.ones(n_minutes)
    for t, evt, scope in zip(sel["t"], sel["event"], sel["scope"]):
        if not (scope == "all" or asset in str(scope).split(",")):
            continue
        wins = table.get(evt, {}).get(grp)
        if not wins:
            continue
        first = int(round((t - pd.Timedelta(minutes=1) - start).total_seconds() / 60.0))
        for w in wins:
            a, b = max(first + w["w0"], 0), min(first + w["w1"], n_minutes)
            if a < b:
                m = max(float(w[p["ev_stat"]]), 0.05) ** p["ev_pow"]
                mult[a:b] = np.maximum(mult[a:b], m) if m >= 1 else np.minimum(mult[a:b], m)
    return mult


_TQ: dict = {}


def _t_quantile(u: np.ndarray, nu: float) -> np.ndarray:
    """Quantiles of Student-t(nu) (nu = 0: standard normal) by interpolation in a logit-spaced
    table built once per nu — exact to ~1e-6 and ~10x faster than scipy's stdtrit per call."""
    if nu not in _TQ:
        from scipy.special import ndtri, stdtrit

        x = np.linspace(-17.0, 17.0, 40001)
        g = 1.0 / (1.0 + np.exp(-x))
        _TQ[nu] = (x, ndtri(g) if nu == 0.0 else stdtrit(nu, g))
    x, q = _TQ[nu]
    uc = np.clip(u, 1e-7, 1 - 1e-7)
    return np.interp(np.log(uc / (1.0 - uc)), x, q)


def _tick(close: np.ndarray) -> float:
    """The price grid step seen in the context (a power of ten), 0 when it cannot be told."""
    d = np.abs(np.diff(close[-5000:]))
    d = d[d > 0]
    if d.size < 50:
        return 0.0
    return float(10.0 ** np.round(np.log10(np.quantile(d, 0.01))))


def _minute_of_week_ny(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert("America/New_York")
    return (ny.day_of_week * 1440 + ny.hour * 60 + ny.minute).to_numpy()


def _weekend(mow: np.ndarray) -> np.ndarray:
    """Friday 17:00 to Sunday 18:00, New York time (minute-of-week index, Monday 00:00 = 0)."""
    return (mow >= 4 * 1440 + 17 * 60) & (mow < 6 * 1440 + 18 * 60)


def _ewma(values: np.ndarray, ages: np.ndarray, half_life: float) -> float:
    w = np.exp2(-ages / half_life)
    s = w.sum()
    return float((w * values).sum() / s) if s > 0 else float("nan")


def _seed(asset: str, start_time, time_length: int) -> int:
    """SYNTH_BENCHMARK_SEED (read once per call), mixed with the prompt's identity so prompts do not
    share draws; a stable crc32, not hash(), so a run is reproducible."""
    import zlib

    raw = os.environ.get("SYNTH_BENCHMARK_SEED")
    try:
        base = int(raw) if raw is not None else 12345
    except ValueError:
        base = 12345
    return (base * 1_000_003 + zlib.crc32(f"{asset}|{start_time}|{time_length}".encode())) % 2**63


def _group(asset: str, time_length: int) -> dict:
    if time_length <= 3600:
        return GROUPS["crypto_1h"]
    return GROUPS["crypto_24h"] if asset in CRYPTO else GROUPS["equity_24h"]


def _params(time_length: int) -> dict:
    p = dict(PARAMS)
    if time_length <= 3600:  # crypto-1h overrides: keys prefixed "1h_"
        p.update({k[3:]: v for k, v in PARAMS.items() if k.startswith("1h_")})
    return p


def _forecast(asset, start_time, time_increment, time_length, context_prices):
    """(last price, per-step log-return variance, minute of week of each step, |ln fast/slow|)."""
    p = _params(time_length)
    t0 = pd.Timestamp(start_time)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    n_steps = time_length // time_increment
    step_min = max(time_increment // 60, 1)
    g = _group(asset, time_length)

    close = context_prices["close"] if isinstance(context_prices, pd.DataFrame) else context_prices
    close = pd.to_numeric(close, errors="coerce")
    vals = close.to_numpy(dtype=float)
    close = close[np.isfinite(vals) & (vals > 0)]
    last = float(close.iloc[-1])

    absest = p["est"] == "abs"
    pk = p["prof_kind"] or ("a" if absest else "v")
    kind = pk + ("1" if step_min == 1 else "5")
    k = 1 if step_min == 1 else 5
    prof = _profile(asset, kind)

    # deseasonalised per-minute variance from the context's k-minute returns (overlapping)
    lp = pd.Series(np.log(close.to_numpy(dtype=float)), index=close.index)
    full = lp.reindex(pd.date_range(lp.index[0], lp.index[-1], freq="1min"))
    v = full.to_numpy()
    rk = v[k:] - v[:-k]
    idx = full.index[:-k]
    hlr = None
    if absest and p["range_w"] > 0 and isinstance(context_prices, pd.DataFrame) and {"high", "low"} <= set(context_prices.columns):
        # range of the k bars a k-min return spans (bars L+1..L+k for the return labelled L)
        hi = pd.to_numeric(context_prices["high"], errors="coerce").reindex(full.index)
        lo = pd.to_numeric(context_prices["low"], errors="coerce").reindex(full.index)
        hmax = hi.rolling(k, min_periods=k).max().to_numpy()[k:]
        lmin = lo.rolling(k, min_periods=k).min().to_numpy()[k:]
        with np.errstate(divide="ignore", invalid="ignore"):
            hlr = np.log(hmax / lmin)
    ok = np.isfinite(rk)
    if hlr is not None:
        ok &= np.isfinite(hlr) & (hlr >= 0)
        hlr = hlr[ok]
    rk, idx = rk[ok], idx[ok]
    hl = p["hl_override"] or g["hl"]
    lv_we = float("nan")
    if rk.size >= 30:
        mow_c = _minute_of_week_ny(idx)
        z = rk**2 / (k * prof[mow_c])
        n_ctx = len(full)
        em = _event_multiplier(asset, full.index[0], n_ctx, p)
        if em is not None:  # remove scheduled-release bursts from the level estimate
            emk = np.convolve(em, np.ones(k) / k, mode="valid")[: n_ctx - k] if k > 1 else em[: n_ctx - k]
            z = z / emk[ok]
        if absest:  # deseasonalised absolute return per sqrt-minute; squared back below
            z = np.sqrt(z)
            if hlr is not None and hlr.size == z.size:
                zr = (hlr / 2.0) / np.sqrt(k * prof[mow_c] * (emk[ok] if em is not None else 1.0))
                c = z.sum() / max(zr.sum(), 1e-300)  # keep the |r| level: the range only cuts noise
                z = (1 - p["range_w"]) * z + p["range_w"] * c * zr
        ages = (full.index[-1] - idx).total_seconds().to_numpy() / 60.0
        ref = _ewma(z, ages, 2880.0)
        z = np.minimum(z, (np.sqrt(p["clip"]) if absest else p["clip"]) * ref)
        lv = np.array([_ewma(z, ages, h) for h in hl])
        we_c = _weekend(mow_c)
        if p["we_alpha"] > 0 and we_c.sum() >= 60:
            lv_we = _ewma(z[we_c], ages[we_c], p["we_hl"])
    else:
        lv = np.full(3, 1e-3 if absest else (1e-3) ** 2)  # ~1bp per minute fallback

    # per-step variance over the horizon
    step_starts = pd.date_range(t0, periods=n_steps, freq=f"{step_min}min")
    mow_f = _minute_of_week_ny(step_starts)
    lead = (np.arange(n_steps) + 0.5) * step_min
    W = np.asarray(g["w"])
    wts = np.stack([np.interp(lead, g["knots"], W[:, j]) for j in range(3)], axis=1)
    if p["we_wts"] is not None and not _weekend(mow_f[:1])[0]:
        # a weekday context's bursts do not carry into the weekend regime
        wts = np.where(_weekend(mow_f)[:, None], np.asarray(p["we_wts"], dtype=float)[None, :], wts)
    level = wts @ lv
    if np.isfinite(lv_we):
        we_f = _weekend(mow_f)
        level = np.where(we_f, p["we_alpha"] * lv_we + (1 - p["we_alpha"]) * level, level)
    if absest:
        level = (np.pi / 2) * p["abs_scale"] * level**2
    if kind[1] == "5" and step_min != 5:
        seas = np.array([prof[(m + np.arange(step_min)) % _WEEK].mean() for m in mow_f])
    else:
        seas = prof[mow_f]
    var = p["scale"] * level * seas * step_min  # per-step variance of the log return
    if p["we_mult"] != 1.0:
        var = np.where(_weekend(mow_f), var * p["we_mult"], var)
    em = _event_multiplier(asset, t0, n_steps * step_min, p)
    if em is not None:
        var = var * em.reshape(n_steps, step_min).mean(axis=1)
    disagree = float(abs(np.log(max(lv[0], 1e-300) / max(lv[2], 1e-300))))
    return last, var, mow_f, disagree


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    p = _params(time_length)
    rng = np.random.default_rng(_seed(asset, start_time, time_length))
    n_steps = time_length // time_increment
    step_min = max(time_increment // 60, 1)
    last, var, mow_f, disagree = _forecast(asset, start_time, time_increment, time_length, context_prices)
    tick = 0.0
    if p["tick_round"]:
        c = context_prices["close"] if isinstance(context_prices, pd.DataFrame) else context_prices
        c = pd.to_numeric(c, errors="coerce").to_numpy(dtype=float)
        tick = _tick(c[np.isfinite(c)])
        if tick / last < 2e-5:  # below 0.2 bp the grid cannot matter
            tick = 0.0

    # scale mixture: per-path variance multiplier with mean 1, Student-t innovations
    sd = float(np.sqrt(p["mix_sd"] ** 2 + (p["mix_state"] * disagree) ** 2))
    nu = p["nu"]
    if p["strat"]:  # Latin hypercube: every step's draws (and the multipliers) cover all N strata
        n = num_simulations
        u = (rng.permutation(n) + rng.random(n)) / n
        mult = np.exp(sd * _t_quantile(u, 0.0) - 0.5 * sd * sd)
        uu = (np.argsort(rng.random((n_steps, n)), axis=1).T + rng.random((n, n_steps))) / n
        eps = _t_quantile(uu, nu) * np.sqrt((nu - 2) / nu)
    else:
        mult = np.exp(sd * rng.standard_normal(num_simulations) - 0.5 * sd * sd)
        eps = rng.standard_t(nu, size=(num_simulations, n_steps)) * np.sqrt((nu - 2) / nu)
    sig = np.sqrt(var)[None, :] * np.sqrt(mult)[:, None]
    sw = p["we_mix_sd"] if asset in CRYPTO else p["we_mix_sd_equ"]
    if sw > 0:  # is this weekend a dead one or an active one? (own per-path draw)
        we = _weekend(mow_f)
        if we.any():
            uw = (rng.permutation(num_simulations) + rng.random(num_simulations)) / num_simulations
            mw = np.exp(sw * _t_quantile(uw, 0.0) - 0.5 * sw * sw)
            sig = np.where(we[None, :], sig * np.sqrt(mw)[:, None], sig)
    if p["mix_rw"] > 0:  # log-variance random walk along the path: level uncertainty grows with lead
        dt = step_min / 1440.0
        inc = p["mix_rw"] * np.sqrt(dt) * rng.standard_normal((num_simulations, n_steps))
        w = np.cumsum(inc, axis=1) - inc  # value at the start of each step
        lead_var = (p["mix_rw"] ** 2) * dt * np.arange(n_steps)
        sig = sig * np.exp(0.5 * (w - 0.5 * lead_var[None, :]))
    if p["sv_sd"] > 0:  # fast stochastic volatility: stationary log-variance AR(1), mean multiplier 1
        phi = 2.0 ** (-1.0 / p["sv_hl"])
        sv = p["sv_sd"]
        eta = rng.standard_normal((num_simulations, n_steps))
        y = np.empty((num_simulations, n_steps))
        y[:, 0] = sv * eta[:, 0]
        c = sv * np.sqrt(1 - phi * phi)
        for j in range(1, n_steps):
            y[:, j] = phi * y[:, j - 1] + c * eta[:, j]
        sig = sig * np.exp(0.5 * (y - 0.5 * sv * sv))
    # exactly-zero steps (no price change), at the profiled share for the step's minute of week;
    # the moving steps carry the variance the zero ones do not, so longer returns keep theirs
    zshare = _profile(asset, "z1" if step_min == 1 else "z5", floor=False)
    zm = p["zero_mult_tick"] if tick > 0 else p["zero_mult"]
    if zshare is not None and zm > 0 and step_min in (1, 5):
        pz = np.clip(zm * zshare[mow_f], 0.0, 0.9)
        sig = sig / np.sqrt(1.0 - pz)[None, :]
        if p["strat"]:  # exactly round(pz * N) stale paths per step
            ranks = np.argsort(rng.random((n_steps, num_simulations)), axis=1).T
            moving = ranks >= np.round(pz * num_simulations)[None, :]
        else:
            moving = rng.random((num_simulations, n_steps)) >= pz[None, :]
        steps = np.where(moving, eps * sig - 0.5 * sig**2, 0.0)
    else:
        steps = eps * sig - 0.5 * sig**2
    logp = np.concatenate([np.zeros((num_simulations, 1)), np.cumsum(steps, axis=1)], axis=1)
    paths = last * np.exp(np.clip(logp, -5.0, 5.0))
    if tick > 0:
        paths = np.maximum(np.round(paths / tick) * tick, tick)
        paths[:, 0] = last
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):  # never answer with an unusable fan
        sig0 = 0.002 * np.sqrt(step_min)
        logp = np.cumsum(rng.normal(0.0, sig0, size=(num_simulations, n_steps)), axis=1)
        paths = last * np.exp(np.concatenate([np.zeros((num_simulations, 1)), logp], axis=1))
    return (start_time, time_increment, *paths)
