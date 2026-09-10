"""Past-only sub-hour scale statistics for an hourly residual ablation."""

import numpy as np
import pandas as pd


def fast_features(context, timestamp, reference):
    close = pd.to_numeric(context.close,errors='coerce')
    close = close.loc[(close.index>timestamp-pd.Timedelta(minutes=60)) & (close.index<=timestamp)]
    close = close.loc[np.isfinite(close) & (close>0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    age = (timestamp-close.index).total_seconds().to_numpy()/60
    gaps = np.diff(close.index.as_unit('ns').asi8)/60e9
    returns = np.diff(np.log(close.to_numpy()))/np.sqrt(gaps)
    values = []
    reference = max(reference,1e-8)
    for width in [5,15,30]:
        selected = returns[age[1:]<width]
        coverage = min(1.,float(np.sum(age<width)/width))
        if len(selected)==0:
            values.extend([0.,0.,0.,coverage])
            continue
        absolute = np.abs(selected)
        estimates = [absolute.mean()*np.sqrt(np.pi/2),np.sqrt(np.mean(selected**2)),absolute.max()]
        values.extend([np.log(max(float(value),1e-9)/reference) for value in estimates])
        values.append(coverage)
    return np.clip(np.nan_to_num(np.asarray(values),nan=0.,posinf=8.,neginf=-8.),-8,8)
