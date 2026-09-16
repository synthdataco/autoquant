"""NumPy inference for small exported histogram-boosted residuals."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def tree_design(static, sequences, timestamps, baseline, forecasts=None):
    summaries = np.concatenate([sequences[:, :, -width:].mean(axis=2)
                                for width in [1, 3, 6, 12, 24, 48, 96, 168]], axis=1)
    timestamps = pd.DatetimeIndex(timestamps)
    day = (timestamps.hour + timestamps.minute / 60) / 24
    week = (timestamps.dayofweek + day) / 7
    clock = np.column_stack([function(2*np.pi*frequency*phase)
                             for phase in [day, week] for frequency in [1, 2]
                             for function in [np.sin, np.cos]])
    designs = {'static': np.column_stack([static, baseline])}
    designs['temporal'] = np.column_stack([designs['static'], summaries, clock])
    if forecasts is not None:
        designs['pretrained'] = np.column_stack([designs['temporal'], forecasts.reshape(len(static), -1)])
    return designs


@lru_cache(maxsize=2)
def tree_weights(kind='tree'):
    return dict(np.load(Path(__file__).with_name(f'{kind}_weights.npz')))


def tree_residual(design, kind='tree'):
    weights = tree_weights(kind)
    output = weights['base'].copy()
    for head in range(2):
        for stage in range(int(weights['stages'])):
            nodes = weights[f'{head}_{stage}']
            index = 0
            while not nodes[index]['is_leaf']:
                node = nodes[index]
                value = design[int(node['feature_idx'])]
                left = bool(node['missing_go_to_left']) if np.isnan(value) else value <= node['num_threshold']
                index = int(node['left'] if left else node['right'])
            output[head] += nodes[index]['value']
    return np.clip(output, -.5, .5)


def tree_residual_batch(design, kind='tree'):
    weights = tree_weights(kind)
    output = np.broadcast_to(weights['base'], (len(design), 2)).copy()
    for head in range(2):
        for stage in range(int(weights['stages'])):
            nodes = weights[f'{head}_{stage}']
            indices = np.zeros(len(design), dtype=int)
            while True:
                active = np.flatnonzero(~nodes[indices]['is_leaf'].astype(bool))
                if not len(active):
                    break
                current = nodes[indices[active]]
                values = design[active, current['feature_idx'].astype(int)]
                left = np.where(np.isnan(values), current['missing_go_to_left'].astype(bool), values <= current['num_threshold'])
                indices[active] = np.where(left, current['left'], current['right'])
            output[:, head] += nodes[indices]['value']
    return np.clip(output, -.5, .5)
