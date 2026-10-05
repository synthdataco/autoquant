"""Model zoo. Every factory returns a callable with the evaluate.py worker signature:

    f(asset, start_time, time_increment, time_length, num_simulations, context_prices, rng)
        -> ndarray (num_simulations, steps)

`agent/modeling.py` wraps the champion factory in the contract the validator expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SIGMA = 1e-6


def _activity_mult(context_prices, step_min, asset, prof_raw, beta, halflives, cap):
    """Bounded width multiplier from `trade_count`: is the tape busier than normal right now?

    Trade count is itself strongly seasonal, so it is divided by its own committed time-of-week
    profile first; the feature is the log ratio of a fast to a slow EWMA of what is left, i.e.
    "unusually busy for this hour" with the hour taken out. A single coefficient and a hard clip,
    rather than a fitted per-asset regression -- the diagnostic in eda_activity.py showed a free
    linear fit extrapolating badly on exactly the thin assets it was meant to help.

    Returns 1.0 whenever the column is missing or too sparse to say anything.
    """
    if "trade_count" not in getattr(context_prices, "columns", ()):
        return 1.0
    s = np.asarray(context_prices["trade_count"], dtype=float)
    idx = np.asarray(context_prices.index.values, dtype="datetime64[m]").astype(np.int64)
    if step_min > 1:
        n = (s.size // step_min) * step_min
        if n < step_min:
            return 1.0
        s = np.nansum(s[:n].reshape(-1, step_min), axis=1)
        take = np.arange(step_min - 1, n, step_min)
        idx = idx[take]
    ok = np.isfinite(s)
    if ok.sum() < 200:
        return 1.0
    prof = prof_raw.get(f"{asset}_act")
    d = s[ok] / (prof[_bidx(idx[ok], prof)] if prof is not None else 1.0)
    lg = np.log1p(np.maximum(d, 0.0))
    ar = np.arange(lg.size - 1, -1, -1)
    out = []
    for h in halflives:
        w = 0.5 ** (ar / float(h))
        out.append(float(np.sum(w * lg) / np.sum(w)))
    a = np.log(max(np.expm1(out[0]), 1e-6)) - np.log(max(np.expm1(out[1]), 1e-6))
    if not np.isfinite(a):
        return 1.0
    return float(np.clip(np.exp(beta * a), 1.0 / cap, cap))


def _step_closes(context_prices, step_min: int):
    """Close series sampled every `step_min` minutes, and the last observed close.

    The context may contain NaN in every column (thin markets, feed gaps). We drop, never
    interpolate: an invented price is a fabricated return, and the scoring already ignores
    unobserved minutes.
    """
    c = context_prices["close"]
    arr = np.asarray(c, dtype=float)
    # Unit-independent minute index. The parquet partitions carry datetime64[us], not [ns], so
    # `.view("int64") // 60e9` silently produced an index that barely moved — a bug that made
    # every return look like it spanned a gap and left the vol estimator with an empty sample.
    idx = np.asarray(c.index.values, dtype="datetime64[m]").astype(np.int64)
    sub = arr[::step_min] if step_min > 1 else arr
    sub_idx = idx[::step_min] if step_min > 1 else idx
    ok = np.isfinite(sub)
    last_all = arr[np.isfinite(arr)]
    last = float(last_all[-1]) if last_all.size else np.nan
    return sub[ok], sub_idx[ok], last


def _log_returns(vals, idx, step_min):
    """Log returns between *consecutive observed* samples, keeping only the ones that span
    exactly one step (a return across a gap is a different random variable)."""
    if vals.size < 3:
        return np.empty(0), np.empty(0, dtype=np.int64)
    r = np.diff(np.log(vals))
    gap = np.diff(idx)
    keep = np.isfinite(r) & (gap == step_min)
    return r[keep], idx[1:][keep]


def _ewma_sigma(r: np.ndarray, halflife: float) -> float:
    if r.size == 0:
        return np.nan
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(r.size - 1, -1, -1)
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def _robust_sd(r: np.ndarray, clip: float = 8.0) -> float:
    """SD after winsorising at `clip` robust sigmas. XRP's raw kurtosis is ~7000: a single
    bad print otherwise sets the whole forecast."""
    if r.size < 5:
        return float(np.std(r)) if r.size else np.nan
    s0 = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    return float(np.std(np.clip(r, -clip * s0, clip * s0)))


# --------------------------------------------------------------------------- baseline


def baseline_lognormal():
    """The harness starter: per-minute sigma from the 7d context, iid Gaussian steps."""

    def f(asset, start_time, time_increment, time_length, num_simulations, context_prices, rng):
        steps = time_length // time_increment + 1
        closes = pd.Series(np.asarray(context_prices["close"], dtype=float)).dropna().to_numpy()
        last = float(closes[-1])
        rets = np.diff(np.log(closes))
        rets = rets[np.isfinite(rets)]
        sigma = float(np.std(rets)) * (time_increment / 60.0) ** 0.5 if rets.size > 10 else 1e-3
        shocks = rng.normal(0.0, sigma, size=(num_simulations, steps))
        shocks[:, 0] = 0.0
        return last * np.exp(np.cumsum(shocks, axis=1))

    return f


# --------------------------------------------------------------------------- iid family


def iid_model(scale: float = 1.0, dist: str = "normal", nu: float = 4.0, halflife: float = 0.0,
              robust: bool = True, pool_days: float = 7.0):
    """i.i.d. steps at the scoring step size. Knobs: scale multiplier, innovation law,
    EWMA half-life of the variance estimate (0 = flat window)."""

    def f(asset, start_time, time_increment, time_length, num_simulations, context_prices, rng):
        steps = time_length // time_increment + 1
        step_min = max(1, time_increment // 60)
        vals, idx, last = _step_closes(context_prices, step_min)
        r, _ = _log_returns(vals, idx, step_min)
        if r.size < 20 or not np.isfinite(last):
            r = np.array([1e-4])
        keep = int(pool_days * 24 * 60 / step_min)
        r = r[-keep:]
        sigma = _ewma_sigma(r, halflife) if halflife > 0 else (_robust_sd(r) if robust else float(np.std(r)))
        sigma = max(float(sigma), MIN_SIGMA) * scale

        if dist == "normal":
            z = rng.standard_normal((num_simulations, steps - 1))
        elif dist == "t":
            z = rng.standard_t(nu, size=(num_simulations, steps - 1)) / np.sqrt(nu / (nu - 2.0))
        elif dist == "empirical":
            s0 = _robust_sd(r) if robust else float(np.std(r))
            pool = r / max(s0, MIN_SIGMA)
            pool = pool - pool.mean()
            z = rng.choice(pool, size=(num_simulations, steps - 1), replace=True)
        else:
            raise ValueError(dist)

        shocks = np.zeros((num_simulations, steps))
        shocks[:, 1:] = sigma * z
        return last * np.exp(np.cumsum(shocks, axis=1))

    return f


# --------------------------------------------------------------------------- seasonal SV family

import seasonality as _seas  # noqa: E402


_WEEK_MIN = 7 * 24 * 60


def _bidx(minutes, prof) -> np.ndarray:
    """Bucket index into `prof` for minute-of-week values.

    The bucket width is read off the profile's own length rather than a module constant, so an
    artifact fitted at a different time-of-week resolution needs no change here.
    """
    return (np.asarray(minutes) % _WEEK_MIN) // (_WEEK_MIN // len(prof))


def _tow_index(ts0, n: int, step_min: int, prof) -> np.ndarray:
    """Time-of-week bucket for each of the n future scoring points starting at ts0."""
    t0 = pd.Timestamp(ts0)
    minute_of_week = (t0.dayofweek * 24 * 60 + t0.hour * 60 + t0.minute) + np.arange(n) * step_min
    return _bidx(minute_of_week, prof)


def _ewma_var_multi(x2: np.ndarray, halflives) -> np.ndarray:
    """EWMA of x2 at several half-lives (in steps), evaluated at the last point."""
    out = []
    n = x2.size
    ar = np.arange(n - 1, -1, -1)
    for hl in halflives:
        w = 0.5 ** (ar / hl)
        out.append(np.sum(w * x2) / np.sum(w))
    return np.array(out)



def _leverage_mult(u: np.ndarray, mode: str, halflife: float, beta: float, cap: float) -> float:
    """Multiplier on the current conditional level from recent downside.

    `u` is the winsorised, deseasonalised return series, so both features are already in units of
    the asset's own volatility and one coefficient can be shared across assets.

      "down": 0.5*log(2*EWMA(u^2 | u<0) / EWMA(u^2)) -- zero under symmetry, positive when the
              recent variance came mostly from down moves. This is the EWMA analogue of the GJR
              asymmetry term.
      "ret" : minus the standardised cumulative return over the same half-life -- the textbook
              leverage effect, positive after a sell-off.
    """
    n = u.size
    if n < 50:
        return 1.0
    w = 0.5 ** (np.arange(n - 1, -1, -1) / halflife)
    sw = np.sum(w)
    if mode == "ret":
        rms = np.sqrt(max(float(np.sum(w * u * u) / sw), 1e-24))
        m = float(np.sum(w * u) / sw)
        # sd of that weighted mean under iid draws of scale `rms`, so x is a z-score
        se = rms * float(np.sqrt(np.sum(w * w))) / sw
        x = float(np.clip(-m / max(se, 1e-24), -4.0, 4.0))
    else:
        u2 = u * u
        allv = float(np.sum(w * u2) / sw)
        downv = float(np.sum(w * u2 * (u < 0)) / sw)
        if allv <= 0 or downv <= 0:
            return 1.0
        x = 0.5 * np.log(2.0 * downv / allv)
        x = float(np.clip(x, -1.0, 1.0))
    return float(np.clip(np.exp(beta * x), 1.0 / cap, cap))


def sv_model(
    profile: str = "tow_pre0701.npz",
    seasonal: bool = True,
    # Profile is raised to this power: 0 = flat, 1 = as measured, >1 = un-shrink it. A dict maps
    # step size in minutes -> exponent, because the 5-minute and 1-minute profiles want opposite
    # corrections (see journal Iter 12); a bare float applies to every step size.
    seas_gamma: float | dict = 1.0,
    # Optional separate exponent for profile values BELOW 1 (quiet buckets). None = symmetric.
    seas_gamma_lo: float | dict | None = None,
    # Width responds to how busy the tape is right now vs normal for this time of week, as a
    # single bounded multiplier exp(beta * a) clipped to [1/act_cap, act_cap]. 0 disables it.
    act_beta: float = 0.0,
    act_halflives=(24.0, 1152.0),   # steps: ~2h and ~4d at the 5-minute step
    act_cap: float = 1.30,
    # Leverage effect: recent downside moves raise the conditional level. 0 disables it.
    # "down" reads the share of recent variance contributed by negative returns; "ret" reads the
    # standardised cumulative return itself. Both are in sd units and enter as exp(lev_beta * x).
    lev_beta: float = 0.0,
    lev_mode: str = "down",
    lev_halflife: float = 288.0,   # steps; 1 day at the 5-minute step
    lev_cap: float = 1.30,
    # Global width level. A dict maps step size in minutes -> level, because the 1-minute step
    # wants a slightly wider one than the 5-minute step (journal Iter 17).
    scale: float | dict = 0.9,
    scale_map=None,   # per-asset multiplier on top of `scale`, fitted out-of-sample
    # Tilts the width linearly across the horizon at constant mean: >0 widens late steps and
    # narrows early ones, <0 the reverse. A pure shape knob -- the ramp is renormalised to mean 1
    # so it cannot move the level, which is swept separately (journal Iter 12). 0 disables it.
    scale_ramp: float = 0.0,
    # Interaction between the calendar and the current regime. The seasonal exponent becomes
    # gamma + seas_gamma_beta * log(v0 / v_long), so a negative beta flattens the time-of-week
    # profile when the market is hot relative to its own week. 0 disables it.
    seas_gamma_beta: float = 0.0,
    # Shrink the reversion target toward a committed per-asset long-run level (vlong.py). The
    # context is only 7 days, so a week that happened to be hot leaves a target that is itself an
    # overshoot; the artifact is the one piece of information about the asset that is NOT in the
    # context. 0 disables it and the target stays the context's own level.
    vlong_art: str | None = None,
    long_blend: float = 0.0,
    long_blend_map=None,            # per-asset override of long_blend, same idea as scale_map
    # Shrink the global tail index toward a per-asset one fitted offline (nufit.py). Same class of
    # information as vlong: the fatness of the innovation is a property of the asset on a timescale
    # longer than the seven-day context. Blended in logs; 0 disables it and `nu` is used as-is.
    nu_art: str | None = None,
    nu_blend: float = 0.0,
    # Per-asset overrides of the two profile exponents, `{asset: {step_min: g}}` (a bare float is
    # accepted and applies at every step). The exponent is the only knob that says how much of an
    # asset's width follows the calendar, and eda_profile.py shows continuously-traded and
    # session-traded assets wanting different answers. Absent asset or step falls back to the
    # global `seas_gamma` / `seas_gamma_lo`.
    seas_gamma_map=None,
    seas_gamma_lo_map=None,
    # Absolute floor on the conditional (deseasonalised) level, in per-step return units.
    # Insurance against a degenerate context, not a modelling choice -- see below. 0 disables it.
    vol_floor: float = 0.0,
    dist: str = "t",
    nu: float = 4.0,
    halflives=(12.0, 72.0, 576.0),
    weights=(0.3, 0.3, 0.4),
    revert_halflife: float = 0.0,   # steps; 0 = no mean reversion toward the long level
    long_halflife: float = 2016.0,
    vol_of_vol: float = 0.0,        # sd of a per-path log-level shock (constant over the path)
    # Persistent within-path vol factor: log-vol follows an AR(1) with this stationary sd and
    # half-life in steps, renormalised so every step's variance is unchanged. Unlike vol_of_vol
    # this is a pure *shape* change -- it fattens the aggregated intervals (30-min, 3-hour,
    # 24-hour and the 1-hour config's *_gaps terms), which a deterministic sigma path makes too
    # close to Gaussian by the CLT.
    vov_ar: float = 0.0,
    vov_map=None,                   # per-asset override of vov_ar, same idea as scale_map
    vov_halflife: float = 72.0,
    # "stationary": the factor starts at its stationary spread, i.e. the current vol level is
    # itself uncertain. "zero": it starts at the level the context implies and diffuses, so the
    # uncertainty grows with horizon. The second is the more defensible prior -- the current
    # level is the one thing the context does pin down -- and is worth testing against the first.
    vov_start: str = "stationary",
    est: str = "rms",               # "rms" (EWMA of u^2) or "mad" (EWMA of |u|)
    level: str = "ewma",            # "ewma" = blended EWMA vols; "fit" = volfit regression
    volfit_art: str = "volfit_pre0701.npz",
    winsor: float = 6.0,
    pool_steps: int = 4032,
):
    """Seasonal stochastic-volatility simulator.

      r_k = sigma_k * z,   sigma_k = scale * v_k * s(t_k)

    `s` is the committed time-of-week profile; `v_k` is a log-linear reversion from the current
    deseasonalised level `v0` (a weighted blend of EWMA variances) toward the long-run level.
    """
    prof_raw = _seas.load(profile)
    fitc = None
    if level == "fit":
        import volfit as _vf0
        fitc = _vf0.load(volfit_art)
    _nufit = None
    if nu_art and nu_blend:
        import nufit as _nf0
        _nufit = _nf0.load(nu_art)

    _vlong = None
    if vlong_art and (long_blend or long_blend_map):
        import vlong as _vl0
        _vlong = _vl0.load(vlong_art)
    _gammas = dict(seas_gamma) if isinstance(seas_gamma, dict) else None
    _gammas_lo = dict(seas_gamma_lo) if isinstance(seas_gamma_lo, dict) else None
    _prof_cache: dict[tuple, dict] = {}

    def _bend(v, g_hi, g_lo):
        """Raise the profile to g_hi above 1 and g_lo below it, then renormalise to mean 1.

        The profile is already mean-normalised, so 1 is the natural hinge: values above it are
        busy buckets and values below are quiet ones. They get separate exponents because the
        shrinkage being undone is not symmetric -- the 0.25 floor in seasonality.py only ever
        binds at the bottom.
        """
        out = np.where(v >= 1.0, v**g_hi, v**g_lo)
        return out / out.mean()

    def _override(m, asset, step_min, dflt):
        if m is None:
            return dflt
        v = m.get(asset)
        if v is None:
            return dflt
        if isinstance(v, dict):
            return float(v.get(step_min, dflt))
        return float(v)

    def _profile_for(step_min: int, asset: str = ""):
        g = float(_gammas.get(step_min, 1.0)) if _gammas is not None else float(seas_gamma)
        if seas_gamma_lo is None:
            g_lo = g
        else:
            g_lo = (float(_gammas_lo.get(step_min, 1.0)) if _gammas_lo is not None
                    else float(seas_gamma_lo))
        g = _override(seas_gamma_map, asset, step_min, g)
        g_lo = _override(seas_gamma_lo_map, asset, step_min,
                         g_lo if seas_gamma_lo is not None else g)
        key = (g, g_lo)
        if key not in _prof_cache:
            _prof_cache[key] = (
                prof_raw if g == 1.0 and g_lo == 1.0
                else {k: _bend(v, g, g_lo) for k, v in prof_raw.items()}
            )
        return _prof_cache[key]
    hl = np.asarray(halflives, dtype=float)
    wts = np.asarray(weights, dtype=float)
    wts = wts / wts.sum()

    def f(asset, start_time, time_increment, time_length, num_simulations, context_prices, rng):
        steps = time_length // time_increment + 1
        step_min = max(1, time_increment // 60)
        vals, idx, last = _step_closes(context_prices, step_min)
        r, rend = _log_returns(vals, idx, step_min)
        if r.size < 30 or not np.isfinite(last):
            sigma_path = np.full(steps - 1, 5e-4)
        else:
            r = r[-pool_steps:]
            rend = rend[-pool_steps:]
            if seasonal:
                key = f"{asset}_{step_min}"
                p = _profile_for(step_min, asset).get(key)
                if p is None:
                    p = np.ones(_seas.N_BUCKETS)
                # bucket of each observed return, from its END minute
                b = _bidx(rend, p)
                s_hist = p[b]
                fut_b = _tow_index(pd.Timestamp(start_time), steps, step_min, p)[1:]
                s_fut = p[fut_b]
            else:
                s_hist = np.ones(r.size)
                s_fut = np.ones(steps - 1)

            u = r / s_hist
            s0 = 1.4826 * np.median(np.abs(u - np.median(u))) + 1e-12
            u = np.clip(u, -winsor * s0, winsor * s0)
            if est == "mad":
                # EWMA of |u| rescaled to a Gaussian sd. For fat-tailed prints the RMS is
                # dominated by a handful of jumps while CRPS is paid on the bulk; the L1
                # estimator tracks the bulk and should need less per-asset shrink.
                u2 = np.abs(u) * np.sqrt(np.pi / 2.0)
                v_multi = np.maximum(_ewma_var_multi(u2, hl), MIN_SIGMA)
            else:
                u2 = u * u
                v_multi = np.sqrt(np.maximum(_ewma_var_multi(u2, hl), MIN_SIGMA**2))
            v0 = float(np.exp(np.sum(wts * np.log(v_multi))))
            if fitc is not None:
                # Same deseasonalised returns, but the level comes from the fitted regression.
                # Features must be built on the RAW (gamma=1) profile the fit was trained on.
                import volfit as _vf1
                C = fitc.get(f"{asset}_{step_min}")
                if C is not None:
                    praw = prof_raw.get(f"{asset}_{step_min}", np.ones(_seas.N_BUCKETS))
                    uf = r / praw[_bidx(rend, praw)]
                    xf = _vf1.features(uf, _vf1.halflives_for(step_min))
                    v0 = float(np.exp((C @ np.concatenate([[1.0], xf]))[-1]))
            if lev_beta:
                v0 *= _leverage_mult(u, lev_mode, lev_halflife, lev_beta, lev_cap)
            # A context in which every close is identical -- a halted or delisted venue, seven
            # days of it -- drives the EWMA down to MIN_SIGMA, which is a numerical guard and not
            # a forecast: the model would serve a distribution ~75x narrower than the quietest
            # real prompt in the snapshot, and CRPS is unbounded above when the truth then moves.
            # The floor caps that loss. It sits well below any level a live asset produces and is
            # verified never to bind (agent/verify_floor.py), so it costs exactly nothing.
            v0 = max(v0, vol_floor)
            if revert_halflife > 0:
                _vl = float(_ewma_var_multi(u2, [long_halflife])[0])
                v_long = max(_vl if est == "mad" else np.sqrt(max(_vl, MIN_SIGMA**2)),
                             MIN_SIGMA, vol_floor)
                if _vlong is not None:
                    lb = (long_blend if long_blend_map is None
                          else float(long_blend_map.get(asset, long_blend)))
                    vb = _vlong.get(f"{asset}_{step_min}")
                    if lb and vb:
                        v_long = float(np.exp((1.0 - lb) * np.log(v_long)
                                              + lb * np.log(vb)))
                k = np.arange(1, steps)
                phi = 0.5 ** (k / revert_halflife)
                v_k = np.exp(np.log(v_long) + phi * (np.log(v0) - np.log(v_long)))
            else:
                v_k = np.full(steps - 1, v0)
            if seas_gamma_beta and seasonal and revert_halflife > 0:
                # Story: a macro shock does not respect trading hours, so in a hot regime the
                # overnight buckets catch up and the day/night ratio compresses. The profile's
                # SHAPE is the only thing that moves -- `_bend` renormalises to mean 1 -- so this
                # cannot smuggle in a level change.
                praw = prof_raw.get(f"{asset}_{step_min}")
                if praw is not None:
                    g0 = (float(_gammas.get(step_min, 1.0)) if _gammas is not None
                          else float(seas_gamma))
                    z = float(np.clip(np.log(v0 / v_long), -1.5, 1.5))
                    ge = float(np.clip(g0 + seas_gamma_beta * z, 0.2, 2.5))
                    s_fut = _bend(praw, ge, ge)[fut_b]
            sigma_path = v_k * s_fut

        sc0 = float(scale.get(step_min, 0.9)) if isinstance(scale, dict) else float(scale)
        sc = sc0 * (1.0 if scale_map is None else float(scale_map.get(asset, 1.0)))
        if act_beta:
            sc *= _activity_mult(context_prices, step_min, asset, prof_raw,
                                 act_beta, act_halflives, act_cap)
        sigma_path = np.maximum(sigma_path, MIN_SIGMA) * sc
        if scale_ramp:
            ramp = 1.0 + scale_ramp * (np.arange(steps - 1) / max(steps - 2, 1))
            sigma_path = sigma_path * (ramp / ramp.mean())

        if dist == "normal":
            z = rng.standard_normal((num_simulations, steps - 1))
        elif dist == "t":
            nu_a = nu
            if _nufit is not None:
                nf = _nufit.get(f"{asset}_{step_min}")
                if nf:
                    nu_a = float(np.clip(np.exp((1.0 - nu_blend) * np.log(nu)
                                                + nu_blend * np.log(nf)), 2.5, 30.0))
            z = (rng.standard_t(nu_a, size=(num_simulations, steps - 1))
                 / np.sqrt(nu_a / (nu_a - 2.0)))
        elif dist == "empirical":
            pool = u / max(float(np.std(u)), MIN_SIGMA)
            pool = pool - pool.mean()
            z = rng.choice(pool, size=(num_simulations, steps - 1), replace=True)
        else:
            raise ValueError(dist)

        sig = np.broadcast_to(sigma_path, (num_simulations, steps - 1))
        if vol_of_vol > 0:
            m = np.exp(rng.normal(-0.5 * vol_of_vol**2, vol_of_vol, size=(num_simulations, 1)))
            sig = sig * m

        vv = vov_ar if vov_map is None else float(vov_map.get(asset, vov_ar))
        if vv > 0 and steps > 2:
            phi = 0.5 ** (1.0 / max(vov_halflife, 1e-9))
            n = steps - 1
            h = np.empty((num_simulations, n))
            zero_start = vov_start == "zero"
            h[:, 0] = 0.0 if zero_start else rng.normal(0.0, vv, size=num_simulations)
            eps = rng.normal(0.0, vv * np.sqrt(1.0 - phi**2), size=(num_simulations, n))
            for k in range(1, n):
                h[:, k] = phi * h[:, k - 1] + eps[:, k]
            # subtract each step's own Var(h_k) so exp(.) has second moment 1 there: the per-step
            # variance is untouched and the factor only redistributes shape across the path
            var_k = (vv**2) * (1.0 - phi ** (2 * np.arange(n))) if zero_start else vv**2
            sig = sig * np.exp(h - var_k)

        shocks = np.zeros((num_simulations, steps))
        shocks[:, 1:] = sig * z
        return last * np.exp(np.cumsum(shocks, axis=1))

    return f


# ------------------------------------------------------- fitted-term-structure SV family


def hvol_model(
    profile: str = "tow_pre0701.npz",
    volfit_art: str = "volfit_pre0701.npz",
    seas_gamma: float = 1.0,
    scale: float = 1.0,
    dist: str = "t",
    nu: float = 4.0,
    vol_of_vol: float = 0.0,
    term: str = "fitted",   # "fitted" = per-step curve from the knots; "flat" = level only
    fallback=None,
):
    """Per-step volatility from a fitted cumulative-variance curve.

    The regression in `volfit.py` predicts, from the same EWMA features, the log RMS of the
    deseasonalised return over the next h steps for h at several knots. Those give a cumulative
    variance curve V(h) = h * exp(2 yhat_h); its increments are the per-step variances, which is
    the term structure that `sv_model`'s `revert_halflife` was hand-approximating. Multiplying
    back by the time-of-week profile of each future step gives sigma_k.
    """
    import volfit as _vf

    prof_raw = _seas.load(profile)
    coefs = _vf.load(volfit_art)
    fb = fallback or sv_model(profile=profile, scale=0.9, revert_halflife=96.0, seas_gamma=0.85)

    def f(asset, start_time, time_increment, time_length, num_simulations, context_prices, rng):
        steps = time_length // time_increment + 1
        step_min = max(1, time_increment // 60)
        key = f"{asset}_{step_min}"
        C = coefs.get(key)
        if C is None:
            return fb(asset, start_time, time_increment, time_length, num_simulations,
                      context_prices, rng)

        vals, idx, last = _step_closes(context_prices, step_min)
        r, rend = _log_returns(vals, idx, step_min)
        pool = _vf.pool_for(step_min)
        hl = _vf.halflives_for(step_min)
        horizons = np.asarray(_vf.horizons_for(step_min), dtype=float)
        p_raw = prof_raw.get(key, np.ones(_seas.N_BUCKETS))

        if r.size < 100 or not np.isfinite(last):
            return fb(asset, start_time, time_increment, time_length, num_simulations,
                      context_prices, rng)

        r = r[-pool:]
        rend = rend[-pool:]
        u = r / p_raw[_bidx(rend, p_raw)]
        x = _vf.features(u, hl)
        A = np.concatenate([[1.0], x])
        yhat = C @ A                                   # log RMS per horizon knot
        V = horizons * np.exp(2.0 * yhat)              # cumulative deseasonalised variance

        k = np.arange(1, steps, dtype=float)
        if term == "flat":
            var_k = np.full(steps - 1, np.exp(2.0 * yhat[-1]))
        else:
            # Log-log interpolation of the cumulative variance, then differentiate. Extrapolate
            # linearly below the first knot rather than letting np.interp clamp -- clamping makes
            # V flat over the first steps, i.e. ZERO dispersion there, which CRPS punishes hard.
            lh, lV = np.log(horizons), np.log(np.maximum(V, 1e-30))
            slope0 = (lV[1] - lV[0]) / (lh[1] - lh[0])
            lk = np.log(k)
            logV = np.where(lk < lh[0], lV[0] + slope0 * (lk - lh[0]), np.interp(lk, lh, lV))
            Vk = np.exp(logV)
            var_k = np.diff(np.concatenate([[0.0], Vk]))
            # a non-monotone V would give a zero-width step; floor at 5% of the mean instead
            var_k = np.maximum(var_k, 0.05 * float(np.mean(np.maximum(var_k, 0.0))) + MIN_SIGMA**2)

        p = p_raw
        if seas_gamma != 1.0:
            p = p**seas_gamma / (p**seas_gamma).mean()
        s_fut = p[_tow_index(pd.Timestamp(start_time), steps, step_min)[1:]]
        sigma_path = np.sqrt(var_k) * s_fut * scale

        if dist == "normal":
            z = rng.standard_normal((num_simulations, steps - 1))
        elif dist == "t":
            z = rng.standard_t(nu, size=(num_simulations, steps - 1)) / np.sqrt(nu / (nu - 2.0))
        elif dist == "empirical":
            s0 = 1.4826 * np.median(np.abs(u - np.median(u))) + 1e-12
            poolz = np.clip(u, -6.0 * s0, 6.0 * s0)
            poolz = poolz / max(float(np.std(poolz)), MIN_SIGMA)
            z = rng.choice(poolz - poolz.mean(), size=(num_simulations, steps - 1), replace=True)
        else:
            raise ValueError(dist)

        sig = np.broadcast_to(sigma_path, (num_simulations, steps - 1))
        if vol_of_vol > 0:
            m = np.exp(rng.normal(-0.5 * vol_of_vol**2, vol_of_vol, size=(num_simulations, 1)))
            sig = sig * m
        shocks = np.zeros((num_simulations, steps))
        shocks[:, 1:] = sig * z
        return last * np.exp(np.cumsum(shocks, axis=1))

    return f
