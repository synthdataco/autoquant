"""Champion candidate v5: EWMA variance + seasonality + two-component normal-mixture
innovations + stratified (quantile-grid) sampling.

Design rationale (see journal.md): the validator scores empirical CRPS on relative
changes in bps at each grid point, summed over intervals. The shortest interval
dominates, so the model puts its effort into a well-calibrated per-step return
distribution:
  - per-step variance: EWMA of squared returns at the scored increment, multiplied by
    an intraday (UTC hour) seasonality factor (median r^2 per hour, shrunk), with mean
    reversion of the level toward the 7-day mean over the forecast horizon;
  - innovations: standardized residuals z = r / sigma_pred are fitted with a
    two-component zero-mean normal mixture (EM) — a peaky calm component plus a wide
    turbulent one. This matches the observed shape (center narrower than any single
    scaled t, tails fatter) while keeping smooth parametric tails, unlike the raw
    empirical grid whose extremes are single context observations;
  - sampling: instead of iid random draws, each step serves the exact quantile grid of
    the innovation distribution under a random per-step permutation. This removes
    sampling noise from the cross-sectional distribution (empirical CRPS is computed
    per time point across paths), a free variance reduction at num_simulations=1000.

Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length//time_increment + 1 points starting at the context's last valid price.
All served points are rounded to 8 significant digits (validator rule:
float(f"{x:.7e}") == x).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

# --- hyperparameters (tuned on backtests — see journal.md) ---
# Overridable via CAMPAIGN_* env vars for tuning sweeps; the champion ships with
# the chosen values hardcoded here.
import os as _os

EWMA_HALFLIFE_DAYS = float(_os.environ.get("CAMPAIGN_EWMA_HL", 1.0))
SEASON_SHRINK = float(_os.environ.get("CAMPAIGN_SEASON_SHRINK", 1.0))
MR_HALFLIFE_DAYS = float(_os.environ.get("CAMPAIGN_MR_HL", 2.0))
VOL_OF_VOL = float(_os.environ.get("CAMPAIGN_VOV", 0.0))  # lognormal sigma of per-path vol multiplier
CALIB_CENTER = int(_os.environ.get("CAMPAIGN_CALIB", 0))  # scale grid by robust std of z
CALIB_LO, CALIB_HI = 0.75, 1.20
SIGMA_MULT = float(_os.environ.get("CAMPAIGN_SIGMA_MULT", 1.0))  # global sigma multiplier
GARCH_ALPHA = float(_os.environ.get("CAMPAIGN_GARCH_ALPHA", 0.15))  # in-simulation vol clustering (0=off)
GARCH_PERSIST = float(_os.environ.get("CAMPAIGN_GARCH_PERSIST", 0.94))  # per 5-min step
FAST_EWMA_HL = float(_os.environ.get("CAMPAIGN_FAST_EWMA", 0.0))  # short EWMA half-life days (0=off)
SPIKE_K = float(_os.environ.get("CAMPAIGN_SPIKE_K", 1.0))  # var_now = max(var_1d, SPIKE_K*var_fast)
MR_TARGET = _os.environ.get("CAMPAIGN_MR_TARGET", "robust")  # "mean" | "robust" (median of daily variances)
DOW_CORRECT = int(_os.environ.get("CAMPAIGN_DOW", 0))  # day-of-week forward/backward vol correction
INNOVATIONS = _os.environ.get("CAMPAIGN_INNOV", "studentt")  # "mixture" | "studentt" | "empirical" | "ml"
ML_WEIGHTS = _os.environ.get("CAMPAIGN_ML_WEIGHTS", str(Path(__file__).resolve().parent / "artifacts" / "mlshape.pt"))
DF_PRIOR = float(_os.environ.get("CAMPAIGN_DF_PRIOR", 4.5))
DF_SHRINK = float(_os.environ.get("CAMPAIGN_DF_SHRINK", 0.6))
DF_MIN, DF_MAX = 3.8, 40.0
MIX_EM_ITERS = 30              # EM iterations for the normal mixture
MIN_RETURNS = 100              # fallback to constant vol below this many grid returns
MIN_FIT = 500                  # min residuals for mixture/empirical fit (else Student-t)
FALLBACK_SIGMA_PER_STEP = 2e-3  # per-step relative vol if context is unusable


def _round_8sig(arr: np.ndarray) -> np.ndarray:
    """Round every point to the validator's 8-significant-digit form."""
    flat = np.asarray(arr, dtype=float).ravel()
    out = np.array([float(f"{v:.7e}") for v in flat], dtype=float)
    return out.reshape(np.asarray(arr).shape)


def _grid_prices(prices: pd.Series, step_seconds: int) -> pd.Series:
    """Last valid close at or before each grid point, grid anchored on the last
    available timestamp (gaps are carried forward, never interpolated)."""
    p = prices.dropna()
    if p.empty:
        return p
    step_min = step_seconds // 60
    end = p.index[-1]
    start = p.index[0].floor(f"{step_min}min")
    grid = pd.date_range(start, end, freq=f"{step_min}min")
    return p.reindex(grid, method="ffill").dropna()


def _ewma(x: np.ndarray, lam: float) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 - lam, adjust=False).mean().to_numpy()


def _seasonality(r2: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Multiplicative hour-of-day (UTC) vol factor, shrunk toward 1 and smoothed.
    Uses the median of squared returns per hour (robust to fat tails)."""
    overall = np.median(r2)
    if not np.isfinite(overall) or overall <= 0:
        return np.ones(24)
    mult = np.ones(24)
    for h in range(24):
        m = r2[hours == h]
        if m.size >= 10:
            med = np.median(m)
            if med > 0:
                mult[h] = (med / overall) ** SEASON_SHRINK
    # circular smoothing [0.25, 0.5, 0.25]
    sm = 0.25 * np.roll(mult, 1) + 0.5 * mult + 0.25 * np.roll(mult, -1)
    return sm / sm.mean()


def _estimate_df(z: np.ndarray) -> float:
    """Student-t df from the excess kurtosis of standardized returns."""
    if z.size < 200:
        return DF_PRIOR
    kurt = float(np.mean(z**4) / max(np.mean(z**2) ** 2, 1e-18) - 3.0)
    df_est = 4.0 + 6.0 / max(kurt, 0.3)
    df_est = float(np.clip(df_est, DF_MIN, DF_MAX))
    return DF_SHRINK * df_est + (1.0 - DF_SHRINK) * DF_PRIOR


def _fit_mixture(z: np.ndarray) -> tuple[float, float, float]:
    """EM for z ~ w*N(0, s1^2) + (1-w)*N(0, s2^2). Returns (w, s1, s2) with s1 <= s2."""
    z = z[np.isfinite(z)]
    # init: split at the median absolute value
    med = np.median(np.abs(z)) + 1e-12
    s1, s2 = med / 0.674, max(float(z.std()), med / 0.674 * 2.0)
    w = 0.8
    for _ in range(MIX_EM_ITERS):
        l1 = w * norm.pdf(z, 0, s1)
        l2 = (1 - w) * norm.pdf(z, 0, s2)
        tot = l1 + l2
        tot = np.maximum(tot, 1e-300)
        p1 = l1 / tot
        w = float(p1.mean())
        s1 = float(np.sqrt(np.sum(p1 * z * z) / max(np.sum(p1), 1e-12)))
        s2 = float(np.sqrt(np.sum((1 - p1) * z * z) / max(np.sum(1 - p1), 1e-12)))
        if s1 > s2:
            s1, s2, w = s2, s1, 1.0 - w
    return w, max(s1, 1e-9), max(s2, 1e-9)


def _mixture_ppf(u: np.ndarray, w: float, s1: float, s2: float) -> np.ndarray:
    """Quantile function of the two-component zero-mean normal mixture (bisection)."""
    lo = np.minimum(norm.ppf(u, 0, s1), norm.ppf(u, 0, s2)) - 1e-9
    hi = np.maximum(norm.ppf(u, 0, s1), norm.ppf(u, 0, s2)) + 1e-9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = w * norm.cdf(mid, 0, s1) + (1 - w) * norm.cdf(mid, 0, s2)
        go_up = cdf < u
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    return 0.5 * (lo + hi)


_ML_CACHE: dict = {}


def _ml_grids(context_prices, time_increment, steps, start_ts, asset, n):
    """Per-step innovation quantile grids from the ML shape model.
    Returns (sigma_1d, grids) with grids shape (steps, n), or None on failure."""
    import torch  # lazy: only the ML branch needs it

    sys_path_added = False
    try:
        from mlshape import QuantileMLP, TAUS, H_GRID, EWMA_HLS_STEPS, ewma as _ml_ewma
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mlshape import QuantileMLP, TAUS, H_GRID, EWMA_HLS_STEPS, ewma as _ml_ewma
        sys_path_added = True

    if "model" not in _ML_CACHE:
        ckpt = torch.load(ML_WEIGHTS, map_location="cpu", weights_only=False)
        _ML_CACHE["ckpt"] = ckpt
    ckpt = _ML_CACHE["ckpt"]
    gp = _grid_prices(context_prices, time_increment)
    r = np.diff(np.log(gp.to_numpy(dtype=float)))
    r = r[np.isfinite(r)]
    hours = gp.index[1:].hour.to_numpy()
    if r.size < 1000:
        return None
    r2 = r**2
    vols = [_ml_ewma(r2, 0.5 ** (1.0 / hl)) for hl in EWMA_HLS_STEPS]
    sig_1d = float(np.sqrt(max(vols[2][-1], 1e-18)))
    lr = [np.log(np.sqrt(max(vols[k][-1], 1e-18)) / sig_1d) for k in (0, 1, 3)]
    z = r / np.sqrt(np.maximum(vols[2], 1e-18))
    z = z[np.isfinite(z)]
    hour = hours[-1]
    import math
    f = np.array([
        np.log(sig_1d * 1e4), lr[0], lr[1], lr[2],
        float(np.mean(np.abs(z[-12:]))), float(np.mean(np.abs(z[-48:]))),
        float(np.max(np.abs(z[-288:]))), float(z[-1]),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
    ], dtype=np.float32)
    is_crypto = 1.0 if asset in ("BTC", "ETH", "SOL", "XRP", "HYPE") else 0.0

    model = QuantileMLP(len(f) + 4, len(TAUS))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mu, sd = ckpt["mu"], ckpt["sd"]
    taus = np.array(ckpt["taus"])
    h_grid = np.array(ckpt["h_grid"])

    # features for each h in H_GRID
    rows = []
    for h in h_grid:
        hh = int((start_ts + pd.Timedelta(seconds=time_increment * int(h))).hour)
        rows.append(np.concatenate([f, [np.log(h),
                                        math.sin(2 * math.pi * hh / 24),
                                        math.cos(2 * math.pi * hh / 24),
                                        is_crypto]]))
    X = (np.array(rows, dtype=np.float32) - mu) / sd
    with torch.no_grad():
        q = model(torch.tensor(X)).numpy()  # (len(h_grid), n_taus)

    # interpolate quantiles over log-h to every step, then to the stratified grid
    u = (np.arange(n) + 0.5) / n
    log_h = np.log(np.arange(1, steps + 1))
    grids = np.empty((steps, n), dtype=np.float64)
    q_logh = np.stack([np.interp(log_h, np.log(h_grid), q[:, k]) for k in range(len(taus))], axis=1)
    for hh in range(steps):
        grids[hh] = np.interp(u, taus, q_logh[hh])
    return sig_1d, grids


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment
    p = context_prices.dropna() if context_prices is not None else pd.Series(dtype=float)
    if p.empty:
        # no context at all: serve a flat fan at price 1 (should never happen)
        paths = np.ones((num_simulations, steps + 1))
        return (start_time, time_increment, *[row.tolist() for row in paths])
    last = float(p.iloc[-1])

    gp = _grid_prices(context_prices, time_increment)
    r = np.diff(np.log(gp.to_numpy(dtype=float)))
    r = r[np.isfinite(r)]
    steps_per_day = 86_400 // time_increment

    if r.size >= MIN_RETURNS:
        lam = 0.5 ** (1.0 / (EWMA_HALFLIFE_DAYS * steps_per_day))
        var_ewma = _ewma(r**2, lam)
        var_now = float(var_ewma[-1])
        if MR_TARGET == "robust":
            # mean-reversion target: median of per-day variances — the 7-day mean of
            # r^2 is tail-inflated, so MR would pull long-horizon levels toward a
            # variance higher than the typical day's
            spd = steps_per_day
            if r.size >= 3 * steps_per_day:
                daily = r[: (r.size // steps_per_day) * steps_per_day].reshape(-1, spd) ** 2
                var_long = float(np.median(daily.mean(axis=1)))
            else:
                var_long = float(np.mean(r**2))
        else:
            var_long = float(np.mean(r**2))
        if FAST_EWMA_HL > 0:
            # spike-activated fast vol: only lifts the level when the short EWMA
            # (scaled by SPIKE_K) exceeds the slow one — adapts to vol-spike onsets
            # without permanently widening calm-regime fans
            lam_f = 0.5 ** (1.0 / (FAST_EWMA_HL * steps_per_day))
            var_now = max(var_now, SPIKE_K * float(_ewma(r**2, lam_f)[-1]))
        hours = gp.index[1:].hour.to_numpy()
        mult = _seasonality(r**2, hours)
        # standardized residuals vs one-step-ahead prediction
        var_pred = np.empty_like(var_ewma)
        var_pred[0] = var_ewma[0]
        var_pred[1:] = var_ewma[:-1]
        z = r / np.sqrt(np.maximum(var_pred * mult[hours], 1e-18))
        z = z[np.isfinite(z)]
        df = _estimate_df(z)
    else:
        var_now = FALLBACK_SIGMA_PER_STEP**2
        var_long = var_now
        mult = np.ones(24)
        z = np.array([])
        df = DF_PRIOR

    # per-step sigma over the horizon: seasonality by UTC hour of each step, and
    # mean reversion of the variance level from the EWMA estimate toward the
    # 7-day mean with half-life MR_HALFLIFE_DAYS
    start_ts = pd.Timestamp(start_time)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    step_hours = np.array(
        [((start_ts + pd.Timedelta(seconds=time_increment * (h + 1))).hour) for h in range(steps)]
    )
    rho = 0.5 ** (1.0 / (MR_HALFLIFE_DAYS * steps_per_day))
    decay = rho ** (np.arange(steps) + 1)
    var_path = mult[step_hours] * (var_long + (var_now - var_long) * decay)
    if DOW_CORRECT and r.size >= MIN_RETURNS:
        # Day-of-week vol correction. The 24h EWMA window does NOT average over
        # days-of-week (unlike hour-of-day, which it averages uniformly), so the
        # EWMA level lags weekly regime changes (Fri->Sat drop). Correct the
        # forward path by the ratio of the forward DOW profile to the
        # EWMA-weighted backward DOW mix. Profile from the context (1 obs/DOW —
        # noisy, but the effect is 2-3x and consistent across assets).
        dow_ctx = gp.index[1:].dayofweek.to_numpy()
        r2 = r**2
        med_all = np.median(r2)
        if np.isfinite(med_all) and med_all > 0:
            profile = np.ones(7)
            for d in range(7):
                m = r2[dow_ctx == d]
                if m.size >= 50:
                    profile[d] = np.median(r2[dow_ctx == d]) / med_all
            profile = profile / profile.mean()
            nb = min(r.size, steps_per_day)
            w = lam ** np.arange(nb)  # j-th most recent obs has weight ∝ lam^j
            w_mix = float(np.sum(w * profile[dow_ctx[r.size - nb :]]) / w.sum())
            step_dows = np.array(
                [(start_ts + pd.Timedelta(seconds=time_increment * (h + 1))).dayofweek
                 for h in range(steps)]
            )
            adj = np.clip(profile[step_dows] / max(w_mix, 1e-9), 0.2, 3.0)
            var_path = var_path * adj
    sigma = np.sqrt(np.maximum(var_path, 1e-18)) * SIGMA_MULT

    # stratified quantile grid of innovations
    n = int(num_simulations)
    u = (np.arange(n) + 0.5) / n
    if INNOVATIONS == "mixture" and z.size >= MIN_FIT:
        w, s1, s2 = _fit_mixture(z)
        z_grid = _mixture_ppf(u, w, s1, s2)
    elif INNOVATIONS == "empirical" and z.size >= MIN_FIT:
        z_grid = np.quantile(z, u)
        z_grid = z_grid / z_grid.std()
    else:
        z_grid = student_t.ppf(u, df)
        z_grid = z_grid / z_grid.std()

    if CALIB_CENTER and z.size >= MIN_FIT:
        # the interquantile center of the standardized residuals is systematically
        # narrower than 1 (the variance forecast is sized by tail-inflated r^2):
        # rescale the served grid so its center matches the realized one
        c = (np.quantile(z, 0.84) - np.quantile(z, 0.16)) / 2.0
        c = float(np.clip(c, CALIB_LO, CALIB_HI))
        z_grid = z_grid * c

    rng = np.random.default_rng()
    if INNOVATIONS == "ml":
        # ML shape model: per-step quantile grids (include vol term structure,
        # seasonality and horizon-dependent tails), scaled by the prompt-time EWMA vol
        ml = None
        try:
            ml = _ml_grids(context_prices, time_increment, steps, start_ts, asset, n)
        except Exception:
            ml = None
        if ml is not None:
            sig_1d, grids = ml
            # use the ML quantiles for SHAPE only (per-step profile over the
            # horizon): de-median and unit-std each row; the level stays with the
            # parametric sigma path (ML conditional levels suffer regression
            # dilution — see journal)
            med = np.median(grids, axis=1, keepdims=True)
            sd_g = grids.std(axis=1, keepdims=True)
            grids = (grids - med) / np.maximum(sd_g, 1e-9)
            logp = np.full(n, np.log(last))
            paths = np.empty((n, steps + 1))
            paths[:, 0] = last
            for h in range(steps):
                logp = logp + sigma[h] * grids[h][rng.permutation(n)]
                paths[:, h + 1] = np.exp(logp)
            paths = _round_8sig(paths)
            return (start_time, time_increment, *[row.tolist() for row in paths])
        # fall through to the parametric model on any failure
    # per-path stochastic vol multiplier (stratified lognormal grid, E[m^2]=1):
    # expresses uncertainty about the vol level itself — high-multiplier paths are
    # high-vol at EVERY step, which fattens long-horizon tails the way vol
    # clustering does in reality
    if VOL_OF_VOL > 0:
        q = norm.ppf(u)
        m = np.exp(VOL_OF_VOL * q)
        m = m / np.sqrt(np.mean(m**2))
        m = m[rng.permutation(n)]
    else:
        m = np.ones(n)
    logp = np.full(n, np.log(last))
    paths = np.empty((n, steps + 1))
    paths[:, 0] = last
    if GARCH_ALPHA > 0:
        # in-simulation vol clustering: per-path variance multiplier follows a
        # GARCH(1,1)-type recursion driven by the path's own squared innovations.
        # Single-step cross-sections stay ~sigma[h]; multi-step sums keep fat tails
        # the way real vol clustering produces them.
        dt = time_increment / 300.0
        alpha = GARCH_ALPHA * dt
        rho = GARCH_PERSIST ** dt
        beta = rho - alpha
        hvar = np.ones(n)
        zvar = float(np.mean(z_grid**2))
        for h in range(steps):
            zh = z_grid[rng.permutation(n)]
            logp = logp + sigma[h] * m * np.sqrt(hvar) * zh
            paths[:, h + 1] = np.exp(logp)
            hvar = 1.0 + alpha * (zh**2 / zvar - 1.0) + beta * (hvar - 1.0)
            hvar = np.maximum(hvar, 0.05)
    else:
        for h in range(steps):
            logp = logp + sigma[h] * m * z_grid[rng.permutation(n)]
            paths[:, h + 1] = np.exp(logp)

    paths = _round_8sig(paths)
    return (start_time, time_increment, *[row.tolist() for row in paths])
