"""Observed-return temporal features and NumPy convolutional inference."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def hourly_observations(close, origin='start_day'):
    close = pd.to_numeric(close, errors='coerce')
    close = close.loc[np.isfinite(close) & (close > 0)].sort_index()
    close = close.loc[~close.index.duplicated(keep='last')]
    gaps = np.diff(close.index.as_unit('ns').asi8) / 60e9
    changes = np.diff(np.log(close.to_numpy())) / np.sqrt(gaps)
    observed = pd.DataFrame({'absolute': np.abs(changes), 'square': changes**2, 'signed': changes}, index=close.index[1:])
    grouped = observed.resample('1h', origin=origin, closed='right', label='right')
    means = grouped.mean()
    means['coverage'] = grouped['absolute'].count() / 60
    return means


def normalize_sequence(hourly, timestamp, reference):
    grid = pd.date_range(end=pd.Timestamp(timestamp).floor('min'), periods=168, freq='1h')
    rows = hourly.reindex(grid)
    absolute = rows.absolute.to_numpy() * np.sqrt(np.pi / 2)
    root_square = np.sqrt(rows.square.to_numpy())
    values = np.stack([np.log(np.maximum(absolute, 1e-9) / reference),
                       np.log(np.maximum(root_square, 1e-9) / reference),
                       rows.signed.to_numpy() / reference,
                       rows.coverage.to_numpy()])
    return np.clip(np.nan_to_num(values, nan=0., posinf=5., neginf=-5.), -5, 5).astype(np.float32)


def context_sequence(context, timestamp, reference):
    close = context.close.loc[(context.index > timestamp - pd.Timedelta(days=7)) & (context.index <= timestamp)]
    hourly = hourly_observations(close, pd.Timestamp(timestamp).floor('min'))
    return normalize_sequence(hourly, timestamp, reference)


@lru_cache(maxsize=4)
def temporal_weights(kind='temporal'):
    return dict(np.load(Path(__file__).with_name(f'{kind}_weights.npz')))


def temporal_embedding(sequence, static, kind='temporal'):
    weights = temporal_weights(kind)
    values = sequence
    for layer in [0, 2, 4]:
        padded = np.pad(values, ((0, 0), (2, 2)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=1)[:, ::2]
        values = np.einsum('ock,ctk->ot', weights[f'encoder.{layer}.weight'], windows) + weights[f'encoder.{layer}.bias'][:, None]
        values = values / (1 + np.exp(-values))
    static = np.clip((static - weights['center']) / weights['spread'], -8, 8)
    values = np.r_[values.reshape(-1), static]
    values = weights['head.0.weight'] @ values + weights['head.0.bias']
    values = values / (1 + np.exp(-values))
    return values


def temporal_residual(sequence, static, kind='temporal'):
    weights = temporal_weights(kind)
    values = temporal_embedding(sequence, static, kind)
    return np.clip(weights['head.2.weight'] @ values + weights['head.2.bias'], -.8, .8)
