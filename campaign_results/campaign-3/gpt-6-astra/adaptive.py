"""Causal calibration from completed, context-local volatility forecasts."""

import numpy as np
import pandas as pd

from modeling import WEIGHTS_AVAILABLE_FROM, _features, _learned_multiplier


def adaptive_ratio(context, timestamp, asset, length):
    timestamp = pd.Timestamp(timestamp)
    close = pd.to_numeric(context.close,errors='coerce')
    close = close.loc[(close.index>timestamp-pd.Timedelta(days=7)) & (close.index<=timestamp)]
    close = close.loc[np.isfinite(close)&(close>0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    if len(close)<50:
        return 1.
    gaps = np.diff(close.index.as_unit('ns').asi8)/60e9
    absolute = pd.Series(np.abs(np.diff(np.log(close.to_numpy())))/np.sqrt(gaps),index=close.index[1:])
    hourly = length<=3600
    horizon,spacing = (60,60) if hourly else (1440,360)
    actual,predicted = [],[]
    for offset in range(9):
        anchor = timestamp.floor('min')-pd.Timedelta(minutes=horizon+offset*spacing)
        if anchor<WEIGHTS_AVAILABLE_FROM:
            continue
        historical = close.loc[close.index<=anchor]
        realized = absolute.loc[(absolute.index>anchor)&(absolute.index<=anchor+pd.Timedelta(minutes=horizon))]
        if len(historical)<50 or len(realized)<max(10,horizon//10):
            continue
        scales,squares = _features(historical,anchor)
        forecast = scales[60 if hourly else 360]*_learned_multiplier(scales,squares,asset,hourly)
        actual.append(float(realized.mean()*np.sqrt(np.pi/2)))
        predicted.append(forecast)
    if len(actual)<3 or sum(predicted)<=0:
        return 1.
    return float(np.clip(sum(actual)/sum(predicted),.5,2.))
