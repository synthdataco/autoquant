"""Past-only quote activity features from the scored close series."""

import numpy as np
import pandas as pd


def activity_features(context, timestamp, reference):
    close = pd.to_numeric(context.close, errors='coerce')
    close = close.loc[(close.index > timestamp-pd.Timedelta(days=7)) & (close.index <= timestamp)]
    close = close.loc[np.isfinite(close) & (close > 0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    age = (timestamp-close.index).total_seconds().to_numpy()/60
    gaps = np.diff(close.index.as_unit('ns').asi8)/60e9
    normalized = np.diff(np.log(close.to_numpy()))/np.sqrt(gaps)
    values = []
    for width in [60, 360, 1440, 10080]:
        selected = age[1:] < width
        returns = normalized[selected]
        intervals = gaps[selected]
        if len(returns) < 2:
            values.extend([0.]*8)
            continue
        absolute = np.abs(returns)
        mean_absolute = max(float(absolute.mean()), 1e-10)
        signs = np.sign(returns)
        sign_correlation = float(np.corrcoef(signs[:-1], signs[1:])[0, 1]) if signs[:-1].std() > 0 and signs[1:].std() > 0 else 0.
        values.extend([float(np.mean(returns == 0)), float(np.sum(age < width)/width),
                       float(np.log1p(intervals.mean())), float(np.mean(intervals > 1)),
                       float(np.log(mean_absolute/max(reference, 1e-8))), sign_correlation,
                       float(np.quantile(absolute, .95)/mean_absolute), float(np.mean(signs))])
    last = float(close.iloc[-1])
    factor = 10.**min(8, max(0, 12-int(np.floor(np.log10(last)))) )
    tick = 0.
    if len(close) > 1 and close.max()*factor < np.iinfo(np.int64).max/4:
        integer_prices = np.rint(close.to_numpy()*factor).astype(np.int64)
        tick = float(np.gcd.reduce(np.abs(np.diff(integer_prices)))/factor)
    values.extend([np.log1p(tick/(last*max(reference, 1e-8))), np.log1p(age[-1])])
    return np.clip(np.nan_to_num(np.asarray(values), nan=0., posinf=8., neginf=-8.), -8, 8)
