"""Final champion model: Asset-and-horizon-specific student-t martingale simulation.

Tuned through empirical hyperparameter optimization:
- Cryptocurrencies: Full 7d window, df_t = 4.0 (1h horizon) or 7.0 (24h horizon).
- Commodities & Equities: 1d lookback window, df_t = 3.0 for heavy-tail coverage.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment + 1
    closes = context_prices["close"].dropna()
    
    # 1. Determine parameters based on asset and horizon
    is_crypto = asset.upper() in ["BTC", "ETH", "SOL", "XRP", "HYPE"]
    
    # Default parameters based on our empirical hyperparameter tuning results
    if is_crypto:
        # Crypto: best performance with full 7d window
        default_window = 10080
        default_df = 4.0 if time_length <= 3600 else 7.0
    else:
        # Equities/Commodities: best performance with 1d window and heavy-tailed DF=3
        default_window = 1440
        default_df = 3.0
        
    # Respect env overrides if present
    window_minutes = os.environ.get("VOL_WINDOW_MINUTES")
    if window_minutes is not None:
        lookback = int(window_minutes)
    else:
        lookback = default_window
        
    # Extract the lookback window
    closes_window = closes.iloc[-lookback:]
    # If the window is too small or thin, fallback to full context
    if len(closes_window) < 100:
        closes_window = closes
        
    last = float(closes_window.iloc[-1])
    rets = np.diff(np.log(np.asarray(closes_window, dtype=float)))
    rets = rets[np.isfinite(rets)]
    
    if rets.size > 10:
        # Compute 1-minute volatility
        halflife_minutes = os.environ.get("VOL_HALFLIFE_MINUTES", None)
        if halflife_minutes is not None:
            rets_series = pd.Series(rets)
            vol_1m = float(rets_series.ewm(halflife=float(halflife_minutes)).std().iloc[-1])
            if not np.isfinite(vol_1m) or vol_1m <= 0:
                vol_1m = float(np.std(rets))
        else:
            vol_1m = float(np.std(rets))
            
        # Scale to step size
        per_step_sigma = vol_1m * (time_increment / 60.0) ** 0.5
    else:
        per_step_sigma = 1e-3
        
    per_step_sigma = max(per_step_sigma, 1e-6)
    
    rng = np.random.default_rng()
    
    # Determine shock distribution
    distribution = os.environ.get("SHOCK_DISTRIBUTION", "student_t").lower()
    if distribution == "student_t":
        df_t = float(os.environ.get("STUDENT_T_DF", default_df))
        scale = ((df_t - 2.0) / df_t) ** 0.5 if df_t > 2.0 else 1.0
        shocks = rng.standard_t(df=df_t, size=(num_simulations, steps)) * (scale * per_step_sigma)
    elif distribution == "laplace":
        scale = 1.0 / (2.0 ** 0.5)
        shocks = rng.laplace(0.0, scale, size=(num_simulations, steps)) * per_step_sigma
    else:
        shocks = rng.normal(0.0, per_step_sigma, size=(num_simulations, steps))
        
    shocks[:, 0] = 0.0
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *[p.tolist() for p in paths])
