"""Causal multiscale volatility sequences with explicit observation fractions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_features import CORE_NAMES, TIME_NAMES, NAMES, one_features
from seasonal_models import get_profile, seasonal_values

SCALES = [(5, 48), (30, 96), (240, 42)]
NUMERIC_NAMES = CORE_NAMES+TIME_NAMES


def normalized_returns(frame, table):
    p = np.asarray(frame.close, dtype=float)
    good = np.isfinite(p) & (p > 0)
    logp = np.full(len(p), np.nan)
    np.log(p, out=logp, where=good)
    r = np.diff(logp, prepend=np.nan)
    gaps = np.asarray((frame.index[1:]-frame.index[:-1]).total_seconds()) != 60
    r[np.r_[False, gaps]] = np.nan
    return r/np.sqrt(seasonal_values(frame.index, table, 'America/New_York'))


def sequence_arrays(returns, bases, endpoints, span_minutes):
    r = np.asarray(returns, dtype=float)
    valid = np.isfinite(r)
    absolute = np.r_[0., np.cumsum(np.where(valid, np.abs(r), 0.))]
    squared = np.r_[0., np.cumsum(np.where(valid, r*r, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    endpoints = np.asarray(endpoints, dtype=int)
    bases = np.maximum(np.asarray(bases, dtype=float), 1e-7)
    lower_context = endpoints-span_minutes+1
    arrays = []
    for width, bins in SCALES:
        high = endpoints[:, None]+1-width*np.arange(bins-1, -1, -1)[None, :]
        low = np.maximum(high-width, lower_context[:, None])
        low = np.clip(low, 0, len(r))
        high = np.maximum(np.clip(high, 0, len(r)), low)
        n = counts[high]-counts[low]
        mean_abs = np.divide(absolute[high]-absolute[low], n,
                             out=np.zeros_like(high, dtype=float), where=n > 0)*np.sqrt(np.pi/2)
        mean_sq = np.divide(squared[high]-squared[low], n,
                            out=np.zeros_like(high, dtype=float), where=n > 0)
        log_abs = np.log(np.maximum(mean_abs/bases[:, None], .01))
        log_rms = np.log(np.maximum(np.sqrt(np.maximum(mean_sq, 0))/bases[:, None], .01))
        log_abs[n == 0] = 0.
        log_rms[n == 0] = 0.
        arrays.append(np.stack([np.clip(log_abs, -5., 3.), np.clip(log_rms, -5., 3.), n/width], axis=-1).astype(np.float32))
    return arrays


def one_sequences(asset, start, horizon, context):
    # Reindexing adds only missing rows. It never estimates an absent price.
    index = pd.date_range(context.index[0], context.index[-1], freq='1min')
    regular = context.reindex(index)
    table = get_profile(asset, str(context.index[-1].date()), 'America/New_York')
    x, base = one_features(asset, start, horizon, context)
    r = normalized_returns(regular, table)
    span = 7*1440-(3 if horizon > 3600 else 2)
    seq = sequence_arrays(r, [base], [len(r)-1], span)
    numeric = x[[NAMES.index(name) for name in NUMERIC_NAMES]].reshape(1, -1)
    numeric[~np.isfinite(numeric)] = 0.
    return seq, numeric
