import numpy as np
import pandas as pd
import scipy.stats

def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment + 1
    last_idx = context_prices.last_valid_index()
    if last_idx is None:
        return (start_time, time_increment, *([[0.0]*steps for _ in range(num_simulations)]))
    last = float(context_prices.loc[last_idx])
    
    # Calculate log returns
    rets = np.diff(np.log(np.asarray(context_prices, dtype=float)))
    rets = rets[np.isfinite(rets)]
    
    if rets.size > 10:
        var_L = np.var(rets)
        
        # Adaptive span depending on the forecasting horizon
        span_mins = 120 if time_length <= 3600 else 1440
        var_S_series = pd.Series(rets).ewm(span=span_mins, min_periods=1).var().values
        valid_vars = var_S_series[np.isfinite(var_S_series)]
        if len(valid_vars) > 0 and valid_vars[-1] > 0:
            var_S = float(valid_vars[-1])
        else:
            var_S = var_L
            
        # Half-life of volatility mean reversion
        half_life_mins = np.clip(time_length / 60.0 / 4.0, 120.0, 720.0)
        kappa = np.log(2) / half_life_mins
        
        times_mins = np.arange(steps) * (time_increment / 60.0)
        expected_var_per_min = var_L + (var_S - var_L) * np.exp(-kappa * times_mins)
        per_step_sigmas = np.sqrt(np.maximum(expected_var_per_min, 1e-12) * (time_increment / 60.0))
        
        # Estimate degrees of freedom dynamically from kurtosis
        kurt = float(scipy.stats.kurtosis(rets, fisher=True))
        if kurt > 0.1:
            df = 4.0 + 6.0 / kurt
            df = float(np.clip(df, 2.5, 30.0))
        else:
            df = 30.0 # effectively normal
            
    else:
        per_step_sigmas = np.full(steps, 1e-3)
        df = 5.0

    rng = np.random.default_rng()
    
    scale = per_step_sigmas * np.sqrt(max(df - 2.0, 0.1) / df)
    shocks = rng.standard_t(df, size=(num_simulations, steps)) * scale[np.newaxis, :]
    shocks[:, 0] = 0.0
    
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    
    return (start_time, time_increment, *[p.tolist() for p in paths])
