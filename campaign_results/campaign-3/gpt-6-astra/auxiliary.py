"""NaN-tolerant, scale-normalized features from optional candle information."""

import numpy as np
import pandas as pd

HALF_LIVES = (30,60,180,360,720,1440,2880,10080)


def auxiliary_features(context,timestamp,reference,absolute):
    age=(pd.Timestamp(timestamp)-context.index).total_seconds().to_numpy()/60
    valid_count=max(int(context.close.notna().sum()),1)
    vector=[]
    usable=False
    if {'high','low'}.issubset(context.columns):
        high=pd.to_numeric(context.high,errors='coerce').to_numpy(dtype=float)
        low=pd.to_numeric(context.low,errors='coerce').to_numpy(dtype=float)
        valid=np.isfinite(high)&np.isfinite(low)&(low>0)&(high>=low)
        ranges=np.log(high[valid]/low[valid])
        usable=bool((ranges>0).sum()>20 and valid.sum()/valid_count>.5)
    else:
        valid=np.zeros(len(context),dtype=bool)
        ranges=np.array([])
    for half_life in HALF_LIVES:
        scale=absolute[half_life]
        if usable:
            weights=np.exp(-np.log(2)*age[valid]/half_life)
            scale=float(np.average(ranges,weights=weights)*np.sqrt(np.pi/8))
        vector.append(np.log(max(scale,1e-9)/max(reference,1e-8)))
    vector.append(min(1.,valid.sum()/valid_count))
    for column in ['volume','trade_count']:
        values=pd.to_numeric(context[column],errors='coerce').to_numpy(dtype=float) if column in context else np.full(len(context),np.nan)
        valid_values=np.isfinite(values)&(values>=0)
        if valid_values.sum()>20 and np.any(values[valid_values]>0):
            means=[]
            for half_life in [60,360,1440,10080]:
                weights=np.exp(-np.log(2)*age[valid_values]/half_life)
                means.append(float(np.average(values[valid_values],weights=weights)))
            vector.extend(np.log((np.asarray(means[:3])+1e-8)/(means[-1]+1e-8)))
            vector.extend([np.log1p(means[-1]),min(1.,valid_values.sum()/valid_count)])
        else:
            vector.extend([0.]*5)
    return np.asarray(vector,dtype=np.float32),usable
