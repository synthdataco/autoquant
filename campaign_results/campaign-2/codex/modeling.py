"""Starter model written by the harness: log-normal martingale calibrated on the context.

This is a deliberately simple STARTING POINT — replace it with your own model.
Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length//time_increment + 1 points starting at the context's last price.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np


def _ewm_std(x: np.ndarray, alpha: float = 0.06) -> float:
    """Exponentially weighted std of 1-minute log returns.

    Alpha≈0.06 (~0.94 daily decay at 1-min sampling) emphasizes recent volatility
    while remaining stable. Falls back to simple std for tiny samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return float(np.std(x)) if n > 1 else 1e-3
    # newest first weights
    w = np.power(1 - alpha, np.arange(n - 1, -1, -1, dtype=float))
    w /= w.sum()
    mean = float(np.sum(w * x))
    var = float(np.sum(w * (x - mean) ** 2))
    return var**0.5


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment + 1
    last = float(context_prices.iloc[-1])
    rets = np.diff(np.log(np.asarray(context_prices, dtype=float)))
    rets = rets[np.isfinite(rets)]
    # Volatility: EWMA of 1m returns
    base_sigma = _ewm_std(rets)

    # Robust dispersion adjuster: align EWMA std to MAD for stability
    if rets.size > 32 and base_sigma > 0:
        mad = float(np.median(np.abs(rets - np.median(rets))))
        mad_std = mad / 0.67448975 if mad > 0 else base_sigma
        robust_adjust = float(np.clip(mad_std / base_sigma, 0.85, 1.15))
    else:
        robust_adjust = 1.0

    # Asset class scaling (light-touch priors)
    asset_upper = str(asset).upper()
    equities = {"SP500", "SPCX", "AAPLX", "GOOGLX", "NVDAX", "TSLAX"}
    commodities = {"XAU", "WTIOIL"}
    if asset_upper in equities:
        asset_mult = 0.96
    elif asset_upper in commodities:
        asset_mult = 0.98
    else:
        asset_mult = 1.00

    # Optional per-asset overrides from agent/config.json
    cfg_path = Path(__file__).resolve().parent / "config.json"
    overrides = {}
    if cfg_path.exists():
        try:
            overrides = json.loads(cfg_path.read_text())
        except Exception:
            overrides = {}
    asset_cfg = overrides.get(asset_upper, {}) if isinstance(overrides, dict) else {}

    # Scale to requested step with all multipliers
    per_step_sigma = base_sigma * robust_adjust * asset_mult * (time_increment / 60.0) ** 0.5
    # Horizon dispersion tweak: slightly tighter for 24h
    if time_length >= 86_400 - 1:
        # global shrink
        per_step_sigma *= 0.90
        # additional per-asset fine-tuning for 24h (overridable)
        long_mult = {
            "SP500": 0.92,
            "AAPLX": 0.90,
            "XAU": 0.95,
            "WTIOIL": 0.95,
        }.get(asset_upper, 1.00)
        long_mult = float(asset_cfg.get("long_sigma_mult", long_mult))
        per_step_sigma *= long_mult
        # Optional regime-aware dispersion (24h only): adjust width based on recent variance
        regime_mode = str(asset_cfg.get("long_regime", "off")).lower()
        if regime_mode in {"variance", "var"} and rets.size > 8:
            hv_minutes = int(float(asset_cfg.get("hv_minutes", 60)))
            hv_minutes = int(np.clip(hv_minutes, 15, 360))
            tail = rets[-hv_minutes:] if rets.size >= hv_minutes else rets
            tail = tail[np.isfinite(tail)]
            if tail.size >= 8 and base_sigma > 0:
                tail_std = float(np.std(tail))
                ratio = tail_std / float(base_sigma)
                low = float(asset_cfg.get("hv_ratio_low", 0.8))
                high = float(asset_cfg.get("hv_ratio_high", 1.2))
                low_mult = float(asset_cfg.get("hv_low_sigma_mult", 0.95))
                high_mult = float(asset_cfg.get("hv_high_sigma_mult", 1.05))
                if ratio < low:
                    per_step_sigma *= low_mult
                elif ratio > high:
                    per_step_sigma *= high_mult
    # Dispersion calibration by horizon (heuristic multipliers)
    if time_length <= 3600 + 1:
        per_step_sigma *= 1.08  # slightly wider for 1h to fit gaps
    elif time_length >= 86_400 - 1:
        per_step_sigma *= 0.92  # slightly tighter for 24h to reduce 24h_abs CRPS
    # Fat tails: Student-t with df from sample kurtosis (bounded for stability)
    # For t-dist: excess_kurtosis = 6/(nu-4) => nu = 6/ek + 4
    if rets.size > 50:
        m2 = float(np.mean((rets - np.mean(rets)) ** 2))
        m4 = float(np.mean((rets - np.mean(rets)) ** 4))
        ek = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0
        nu_est = 6.0 / max(ek, 1e-6) + 4.0 if ek > 0 else 12.0
    else:
        nu_est = 12.0

    # Horizon-aware tails: favor fatter tails for 1h, near-normal for 24h
    short_h = time_length <= 3600 + 1
    long_h = time_length >= 86_400 - 1
    if short_h:
        # encourage fat tails; Student-t, fixed mixture, or adaptive mixture
        rng = np.random.default_rng()
        short_mode = str(asset_cfg.get("short_shock", "t")).lower()
        if short_mode == "mixgauss":
            p = float(asset_cfg.get("mix_p", 0.05))
            r = float(asset_cfg.get("mix_r", 4.0))  # sigma_big / sigma_small
            s1 = per_step_sigma / (((1 - p) + p * (r ** 2)) ** 0.5)
            s2 = r * s1
            shocks = rng.normal(0.0, s1, size=(num_simulations, steps))
            big_mask = rng.random(size=(num_simulations, steps)) < p
            shocks[big_mask] = rng.normal(0.0, s2, size=big_mask.sum())
        elif short_mode == "mixauto":
            # Choose p to match sample excess kurtosis via bisection
            r = float(asset_cfg.get("mix_r", 4.0))
            p_cap = float(asset_cfg.get("mix_p_cap", 0.2))
            p_cap = float(np.clip(p_cap, 0.02, 0.3))
            # target excess kurtosis from returns (bounded)
            if rets.size > 50:
                m2 = float(np.mean((rets - np.mean(rets)) ** 2))
                m4 = float(np.mean((rets - np.mean(rets)) ** 4))
                ek_t = max(0.0, min(10.0, (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0))
            else:
                ek_t = 2.0

            def ek_mix(p: float) -> float:
                v = (1 - p) + p * (r ** 2)
                num = (1 - p) + p * (r ** 4)
                return 3.0 * (num / (v * v)) - 3.0

            lo, hi = 0.001, p_cap
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if ek_mix(mid) < ek_t:
                    lo = mid
                else:
                    hi = mid
            p = float((lo + hi) * 0.5)
            s1 = per_step_sigma / (((1 - p) + p * (r ** 2)) ** 0.5)
            s2 = r * s1
            shocks = rng.normal(0.0, s1, size=(num_simulations, steps))
            big_mask = rng.random(size=(num_simulations, steps)) < p
            shocks[big_mask] = rng.normal(0.0, s2, size=big_mask.sum())
        else:
            nu = float(np.clip(nu_est, 6.0, 18.0))
            var_t = nu / (nu - 2.0)
            scale = per_step_sigma / (var_t**0.5)
            shocks = rng.standard_t(nu, size=(num_simulations, steps)) * scale

        # Optional early-minutes per-step multipliers to shape 1h dispersion.
        # Config keys (per asset in agent/config.json):
        # - short_minute_mults: list of floats length==steps; applied elementwise to shocks
        # - OR legacy: short_sigma_boost (float), short_tau_minutes (float) for exp-decay boost
        mm = asset_cfg.get("short_minute_mults")
        if isinstance(mm, list) and len(mm) == int(steps):
            try:
                mult = np.asarray(mm, dtype=float)
                mult = np.clip(mult, 0.5, 2.0)
                shocks = shocks * mult
            except Exception:
                pass
        else:
            # Legacy exponential boost path
            boost = float(asset_cfg.get("short_sigma_boost", 0.0))
            tau = float(asset_cfg.get("short_tau_minutes", 10.0))
            if boost != 0.0 and tau > 0.0:
                t_idx = np.arange(steps, dtype=float)
                # per-step multiplier decays from (1+boost) at k=1 toward 1.0
                mult = 1.0 + boost * np.exp(-(t_idx) / tau)
                shocks = shocks * mult
    elif long_h:
        # 24h: Gaussian (default) or variance-mixture shocks + weak mean-reversion (discrete OU on log-price)
        rng = np.random.default_rng()
        # Long-run mean anchor: default 12h EMA; optional multi-scale EMA blend via config.
        ctx = np.asarray(context_prices, dtype=float)
        ctx = ctx[np.isfinite(ctx)]
        log_ctx = np.log(ctx + 1e-12)

        def ema_tail(x: np.ndarray, minutes: int) -> float:
            n = min(minutes, x.size)
            if n <= 1:
                return float(x[-1]) if x.size else 0.0
            a = 2.0 / (n + 1.0)
            m_ = x[-n]
            for vv in x[-n + 1 :]:
                m_ = a * vv + (1 - a) * m_
            return float(m_)

        anchor_mode = str(asset_cfg.get("long_anchor", "ema12")).lower()
        if anchor_mode == "blend":
            m6 = ema_tail(log_ctx, 360)
            m12 = ema_tail(log_ctx, 720)
            m24 = ema_tail(log_ctx, 1440)
            # Heuristic weights favoring shorter memory but with some longer-term pull
            mu_long = 0.50 * m6 + 0.35 * m12 + 0.15 * m24
        elif anchor_mode == "trend":
            # EMA anchor plus additive linear-trend drift estimated over recent window
            m12 = ema_tail(log_ctx, 720)
            mu_long = m12
            # robust linear fit on last T minutes (default 720)
            T = int(float(asset_cfg.get("trend_minutes", 720)))
            T = int(np.clip(T, 120, 2160))
            tail = log_ctx[-min(T, log_ctx.size):]
            if tail.size >= 16:
                # Regress tail against time index in hours for numerical stability
                t = np.arange(tail.size, dtype=float) / 60.0  # hours
                t = t - t.mean()
                y = tail - tail.mean()
                denom = float((t * t).sum())
                slope_h = float((t * y).sum() / denom) if denom > 0 else 0.0  # per hour
                drift_scale = float(asset_cfg.get("trend_scale", 0.5))
                trend_per_hour = drift_scale * slope_h
            else:
                trend_per_hour = 0.0
        else:
            mu_long = ema_tail(log_ctx, 720)  # ~12h

        # OU: X_{t+dt} = X_t + theta*(mu_long - X_t)*dt + sigma*sqrt(dt)*eps
        dt_hours = time_increment / 3600.0
        # per-hour mean reversion strength (weak); overridable via env var `OU_THETA`
        # Asset-specific default theta with env override
        asset_upper = str(asset).upper()
        default_theta_map = {
            "SP500": 0.04,
            "AAPLX": 0.05,
            "XAU": 0.03,
            "WTIOIL": 0.03,
            "BTC": 0.02,
            "ETH": 0.03,
            "SOL": 0.03,
        }
        default_theta = default_theta_map.get(asset_upper, 0.03)
        if isinstance(asset_cfg, dict) and "theta" in asset_cfg:
            try:
                default_theta = float(asset_cfg["theta"])
            except Exception:
                pass
        try:
            theta = float(os.getenv("OU_THETA", str(default_theta)))
        except Exception:
            theta = default_theta
        theta = float(np.clip(theta, 0.005, 0.2))
        x0 = float(np.log(last + 1e-12))
        # Optional long-horizon shock family (default: Gaussian).
        # - 'vg': Normal with Gamma variance mixing (mean=1), ek controlled by shape k.
        # - 'nig': symmetric Normal-Inverse-Gaussian via Normal with IG mixing; ek set by α.
        long_shock = str(asset_cfg.get("long_shock", "gauss")).lower()

        X = np.full((num_simulations,), x0, dtype=float)
        paths = np.empty((num_simulations, steps), dtype=float)
        paths[:, 0] = last
        if long_shock == "vg":
            # Target excess kurtosis from returns (capped small for 24h stability)
            if rets.size > 200:
                m2 = float(np.mean((rets - np.mean(rets)) ** 2))
                m4 = float(np.mean((rets - np.mean(rets)) ** 4))
                ek_t = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0
                ek_t = float(np.clip(ek_t, 0.0, float(asset_cfg.get("long_ek_cap", 0.4))))
            else:
                ek_t = float(asset_cfg.get("long_ek_default", 0.1))
            # For scale mixture N(0, sigma^2 G) with E[G]=1 and Var[G]=1/k (Gamma k, scale=1/k),
            # excess kurtosis = 3 Var[G] = 3/k  => k = 3 / ek.
            k_shape = 3.0 / max(ek_t, 1e-6)
            # Sample per-step mixing G ~ Gamma(k, scale=1/k), i.i.d. across steps/paths.
            for i in range(1, steps):
                G = rng.gamma(shape=k_shape, scale=1.0 / k_shape, size=num_simulations)
                eps = rng.normal(0.0, 1.0, size=num_simulations) * np.sqrt(G)
                if anchor_mode == "trend":
                    X = X + theta * (mu_long - X) * dt_hours + trend_per_hour * dt_hours + per_step_sigma * eps
                else:
                    X = X + theta * (mu_long - X) * dt_hours + per_step_sigma * eps
                paths[:, i] = np.exp(X)
        elif long_shock == "nig":
            # Symmetric NIG: variance mixture with Inverse-Gaussian mixing V ~ IG(μ=1, λ=α^2)
            # Choose α from target excess kurtosis ek: for symmetric NIG with Var=1, ek = 3/α^2.
            if rets.size > 200:
                m2 = float(np.mean((rets - np.mean(rets)) ** 2))
                m4 = float(np.mean((rets - np.mean(rets)) ** 4))
                ek_t = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0
                ek_t = float(np.clip(ek_t, 0.0, float(asset_cfg.get("long_ek_cap", 0.4))))
            else:
                ek_t = float(asset_cfg.get("long_ek_default", 0.1))
            ek_t = max(ek_t, 1e-6)
            alpha = float(np.sqrt(3.0 / ek_t))

            def sample_ig(mu: float, lam: float, size: int) -> np.ndarray:
                # Michael-Schucany-Haas sampler
                v = rng.normal(0.0, 1.0, size)
                y = v * v
                x = mu + (mu * mu * y) / (2.0 * lam) - (mu / (2.0 * lam)) * np.sqrt(4.0 * mu * lam * y + mu * mu * y * y)
                u = rng.random(size)
                cond = u <= (mu / (mu + x))
                out = np.where(cond, x, (mu * mu) / x)
                return out

            for i in range(1, steps):
                V = sample_ig(1.0, alpha * alpha, num_simulations)  # mean 1
                eps = rng.normal(0.0, 1.0, size=num_simulations) * np.sqrt(V)
                if anchor_mode == "trend":
                    X = X + theta * (mu_long - X) * dt_hours + trend_per_hour * dt_hours + per_step_sigma * eps
                else:
                    X = X + theta * (mu_long - X) * dt_hours + per_step_sigma * eps
                paths[:, i] = np.exp(X)
        else:
            for k in range(1, steps):
                eps = rng.normal(0.0, 1.0, size=num_simulations)
                if anchor_mode == "trend":
                    X = X + theta * (mu_long - X) * dt_hours + trend_per_hour * dt_hours + per_step_sigma * eps
                else:
                    X = X + theta * (mu_long - X) * dt_hours + per_step_sigma * eps
                paths[:, k] = np.exp(X)
        return (start_time, time_increment, *[p.tolist() for p in paths])
    else:
        # mid horizons: blend
        nu = float(np.clip(nu_est, 10.0, 24.0))
        var_t = nu / (nu - 2.0)
        scale = per_step_sigma / (var_t**0.5)
        rng = np.random.default_rng()
        shocks = rng.standard_t(nu, size=(num_simulations, steps)) * scale
    shocks[:, 0] = 0.0
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *[p.tolist() for p in paths])
