"""Smooth weekly calendar features for mean-preserving lead forecasts."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def weekly_design(timestamps, assets, edges):
    timestamps = pd.DatetimeIndex(timestamps)
    phase = (timestamps.dayofweek+(timestamps.hour+timestamps.minute/60)/24)/7
    duration = np.diff(edges)/edges[-1]
    features = []
    for frequency in [1, 2]:
        angular = 2*np.pi*frequency
        left = angular*(np.asarray(phase)[:,None]+edges[:-1]/10080)
        right = angular*(np.asarray(phase)[:,None]+edges[1:]/10080)
        features.extend([(np.cos(left)-np.cos(right))/(right-left),
                         (np.sin(right)-np.sin(left))/(right-left)])
    harmonic = np.stack(features,axis=2)
    harmonic -= (harmonic*duration[None,:,None]).sum(axis=1,keepdims=True)
    identity = np.eye(13)[np.asarray(assets)]
    interactions = (harmonic[:,:,:,None]*identity[:,None,None,:]).reshape(len(timestamps),5,-1)
    return np.concatenate([harmonic,interactions],axis=2)


@lru_cache(maxsize=1)
def weekly_weights():
    with np.load(Path(__file__).with_name('weekly_lead_weights.npz')) as archive:
        return dict(archive)


def weekly_buckets(buckets, timestamp, asset_index):
    weights = weekly_weights()
    design = weekly_design([timestamp],[asset_index],weights['edges'])[0]
    correction = np.clip(design @ weights['coefficients'],-.25,.25)
    adjusted = buckets*np.exp(correction)
    duration = weights['duration']
    return adjusted*(buckets @ duration)/(adjusted @ duration)
