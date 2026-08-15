from __future__ import annotations
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
    
    # Handle both Series and DataFrame context_prices safely
    if isinstance(context_prices, pd.DataFrame):
        prices = np.asarray(context_prices["close"], dtype=float)
    else:
        prices = np.asarray(context_prices, dtype=float)
        
    last = float(prices[-1])
    rets = np.diff(np.log(prices))
    rets = rets[np.isfinite(rets)]
    
    if time_length == 3600:
        if rets.size > 10:
            rets_series = pd.Series(rets)
            ewma_std = rets_series.ewm(span=120, min_periods=10).std().iloc[-1]
            per_step_sigma = float(ewma_std) * (time_increment / 60.0) ** 0.5
        else:
            per_step_sigma = 1e-3
        df = 4.0
    else:
        per_step_sigma = float(np.std(rets)) * (time_increment / 60.0) ** 0.5 if rets.size > 10 else 1e-3
        
        crypto_assets = {"BTC", "ETH", "SOL", "XRP", "HYPE"}
        if asset in crypto_assets:
            per_step_sigma *= 0.85
        else:
            per_step_sigma *= 0.50
            
        df = None
        
    rng = np.random.default_rng()
    
    if df is not None:
        scale = per_step_sigma * np.sqrt((df - 2.0) / df)
        shocks = rng.standard_t(df, size=(num_simulations, steps)) * scale
    else:
        shocks = rng.normal(0.0, per_step_sigma, size=(num_simulations, steps))
        
    shocks[:, 0] = 0.0
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    
    return (start_time, time_increment, *[p.tolist() for p in paths])
