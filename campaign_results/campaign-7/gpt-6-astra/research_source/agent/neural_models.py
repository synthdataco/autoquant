"""Frozen CPU inference for the multiscale volatility experiment."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
import torch

from ml_models import scale_adjustment
from neural_features import one_sequences
from seasonal_models import forecast_variance

CONTEXT_DAYS = 7
VARIANT = os.environ.get('SYNTH_RESEARCH_VARIANT', 'neural_blend')
ART = Path(__file__).resolve().parent/'artifacts/neural'
torch.set_num_threads(1)


@lru_cache(maxsize=1)
def manifest():
    return json.loads((ART/'manifest.json').read_text())


@lru_cache(maxsize=16)
def model(filename):
    # Frozen TorchScript loads complete parameters without random initialization.
    return torch.jit.load(str(ART/filename), map_location='cpu').eval()


def neural_adjustment(asset, start, horizon, context):
    meta = manifest()
    if asset not in meta['assets']:
        return 1.
    asof = str(context.index[-1].date())
    fits = [f for f in meta['fits'] if f['cutoff'] <= asof and f['time_length'] == horizon]
    if not fits:
        return 1.
    fit = max(fits, key=lambda f: f['cutoff'])
    seq, numeric = one_sequences(asset, start, horizon, context)
    args = [torch.as_tensor(x) for x in [*seq, numeric]]
    args.append(torch.tensor([meta['assets'].index(asset)], dtype=torch.long))
    with torch.inference_mode():
        value = float(model(fit['file'])(*args)[0])
    return float(np.exp(np.clip(value, np.log(.25), np.log(3.))))


def simulate(asset, start_time, time_increment=300, time_length=86400,
             num_simulations=1000, context_prices=None):
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    rng = np.random.default_rng(seed)
    last, stepvar = forecast_variance(asset, start_time, time_increment, time_length, context_prices)
    neural = neural_adjustment(asset, start_time, time_length, context_prices)
    if VARIANT == 'neural_blend':
        original = scale_adjustment(asset, start_time, time_length, context_prices, 'vol', 1.)
        neural = np.sqrt(neural*original)
    z = rng.normal(size=(num_simulations, len(stepvar)+1))
    z[:, 0] = 0.
    z[:, 1:] *= np.sqrt(stepvar)*neural
    paths = last*np.exp(np.cumsum(z, axis=1))
    return (start_time, time_increment, *paths.tolist())
