"""Chronologically selected ridge location models, with zero benchmark."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from location import HORIZONS, LAGS
from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from train_vol import ASSETS


def make_asset(asset):
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    indices = np.flatnonzero(np.argmax(data['inputs'][:, -len(ASSETS):], axis=1) == ASSETS.index(asset))
    static = data['inputs'][indices]
    timestamps = pd.to_datetime(data['timestamps'][indices], utc=True)
    cutoff = pd.Timestamp('2026-07-26', tz='UTC')
    prices = load_context(asset, timestamps.min() - pd.Timedelta(days=7), cutoff)
    close = prices.close.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq='1min'))
    logs = np.log(close.where(np.isfinite(close) & (close > 0)))
    current = logs.reindex(timestamps).to_numpy()
    reference = np.exp(static[:, 16])
    signed = np.stack([(current - logs.shift(int(lag)).reindex(timestamps).to_numpy()) / (reference * np.sqrt(lag)) for lag in LAGS], axis=1)
    observed = np.isfinite(signed)
    inputs = np.column_stack([static, np.clip(np.nan_to_num(signed, nan=0.), -8, 8), observed])
    targets = np.stack([(logs.shift(-int(horizon)).reindex(timestamps).to_numpy() - current) / (reference * np.sqrt(horizon)) for horizon in HORIZONS], axis=1)
    valid = np.isfinite(targets).all(axis=1) & (timestamps + pd.Timedelta(days=1) < cutoff)
    print(asset, int(valid.sum()), flush=True)
    return inputs[valid], targets[valid], timestamps.as_unit('ns').asi8[valid]


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    with ProcessPoolExecutor(max_workers=2) as executor:
        batches = list(executor.map(make_asset, ASSETS))
    inputs = np.concatenate([batch[0] for batch in batches])
    targets = np.clip(np.concatenate([batch[1] for batch in batches]), -10, 10)
    timestamps = np.concatenate([batch[2] for batch in batches])
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC').value
    center = inputs[training].mean(axis=0)
    spread = np.maximum(inputs[training].std(axis=0), .1)
    design = np.column_stack([np.ones(len(inputs)), np.clip((inputs - center) / spread, -8, 8)])
    baseline = float(np.mean(targets[~training]**2))
    best, selected_penalty = baseline, None
    coefficients = np.zeros((design.shape[1], targets.shape[1]))
    diagnostics = {}
    for penalty in [100, 1000, 10000]:
        fitted = np.linalg.solve(design[training].T @ design[training] + np.eye(design.shape[1]) * penalty,
                                 design[training].T @ targets[training])
        prediction = np.clip(design[~training] @ fitted, -.3, .3)
        loss = float(np.mean((prediction - targets[~training])**2))
        diagnostics[str(penalty)] = loss
        if loss < best:
            best, selected_penalty, coefficients = loss, penalty, fitted
    np.savez(AGENT_DIR / 'location_weights.npz', center=center, spread=spread, coefficients=coefficients)
    metrics = {'baseline_validation_mse': baseline, 'best_validation_mse': best, 'penalty': selected_penalty,
               'diagnostics': diagnostics, 'training_rows': int(training.sum()), 'validation_rows': int((~training).sum()),
               'target_cutoff': '2026-07-26', 'validation_start': '2026-07-01'}
    (AGENT_DIR / 'location_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
