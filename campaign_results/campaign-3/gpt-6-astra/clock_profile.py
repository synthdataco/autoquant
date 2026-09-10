"""Strictly positive empirical variance clocks at different resolutions."""

import numpy as np
import pandas as pd


def estimate_profile(context, timestamp, bins, floor):
    close = pd.to_numeric(context.close,errors='coerce')
    close = close.loc[(close.index>timestamp-pd.Timedelta(days=7)) & (close.index<=timestamp)]
    close = close.loc[np.isfinite(close)&(close>0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    if len(close)<2:
        return np.ones(bins)
    gaps = np.diff(close.index.as_unit('ns').asi8)/60e9
    absolute = np.abs(np.diff(np.log(close.to_numpy())))/np.sqrt(gaps)
    minute = close.index[1:].hour*60+close.index[1:].minute
    clock = pd.DataFrame({'absolute':absolute,'bin':minute*bins//1440,'day':close.index[1:].date})
    daily_mean = clock.groupby('day').absolute.transform('mean').clip(lower=1e-9)
    clock['relative'] = clock.absolute/daily_mean
    profile = clock.groupby('bin').relative.mean().reindex(range(bins),fill_value=1).to_numpy()
    profile = .25*np.roll(profile,1)+.5*profile+.25*np.roll(profile,-1)
    profile = np.clip(profile,floor,3)
    return profile/profile.mean()
