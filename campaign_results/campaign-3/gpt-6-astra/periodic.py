"""Smooth positive scale corrections, without market-session assumptions."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def periodic_design(timestamps, asset_indices, asset_count=13):
    timestamps = pd.DatetimeIndex(timestamps)
    day = (timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600) / 24
    week = (timestamps.dayofweek + day) / 7
    harmonic = np.column_stack([function(2*np.pi*frequency*phase)
                               for phase in [day,week] for frequency in [1,2] for function in [np.sin,np.cos]])
    indicator = np.eye(asset_count)[asset_indices]
    interactions = (harmonic[:, :, None] * indicator[:, None, :]).reshape(len(timestamps), -1)
    return np.column_stack([harmonic,interactions])


@lru_cache(maxsize=1)
def periodic_weights():
    return dict(np.load(Path(__file__).with_name('periodic_weights.npz')))


def periodic_multiplier(asset_index, timestamp, length):
    design = periodic_design(pd.DatetimeIndex([timestamp]),np.array([asset_index]))
    output = design @ periodic_weights()['coefficients']
    return float(np.exp(np.clip(output[0,0 if length==3600 else 1],-.5,.5)))
