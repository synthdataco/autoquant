"""Change only daily supervised labels to match five-minute increments."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from train_vol import ASSETS, train


def make_asset(asset):
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    selected = np.argmax(data['inputs'][:, -len(ASSETS):], axis=1) == ASSETS.index(asset)
    inputs, targets = data['inputs'][selected], data['targets'][selected].copy()
    timestamps = pd.to_datetime(data['timestamps'][selected], utc=True)
    cutoff = pd.Timestamp('2026-07-26', tz='UTC')
    prices = load_context(asset, timestamps.min(), cutoff)
    close = prices.close.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq='1min'))
    close = close.where(np.isfinite(close) & (close > 0))
    forward = np.abs(np.log(close.shift(-5) / close)) / np.sqrt(5)
    future = forward.iloc[::-1].rolling(1436, min_periods=144).mean().iloc[::-1]
    targets[:, 1] = future.reindex(timestamps).to_numpy() * np.sqrt(np.pi / 2) / np.exp(inputs[:, 16])
    valid = np.isfinite(targets).all(axis=1) & (targets > 0).all(axis=1)
    valid &= timestamps + pd.Timedelta(days=1) < cutoff
    print(asset, int(valid.sum()), 'median daily target ratio', np.median(targets[valid, 1] / data['targets'][selected][valid, 1]), flush=True)
    return inputs[valid], targets[valid], timestamps.as_unit('ns').asi8[valid]


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    dataset = AGENT_DIR / 'results' / 'fiveminute_training.npz'
    if not dataset.exists():
        with ProcessPoolExecutor(max_workers=2) as executor:
            batches = list(executor.map(make_asset, ASSETS))
        np.savez(dataset, inputs=np.concatenate([batch[0] for batch in batches]),
                 targets=np.concatenate([batch[1] for batch in batches]),
                 timestamps=np.concatenate([batch[2] for batch in batches]))
    train(dataset, AGENT_DIR / 'fiveminute_weights.npz')


if __name__ == '__main__':
    main()
