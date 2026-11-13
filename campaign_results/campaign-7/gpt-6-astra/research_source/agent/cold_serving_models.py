"""Serve the fixed checkpoint with three real days and leading NaN days."""
import numpy as np
import pandas as pd
from combined_checkpoint import simulate as serve

CONTEXT_DAYS = 7


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    context = context_prices.copy(deep=True)
    context.loc[context.index < pd.Timestamp(start_time)-pd.Timedelta(days=3), :] = np.nan
    return serve(asset, start_time, time_increment, time_length,
                 num_simulations, context)
