"""Chronologically trained pooled volatility forecasters, without price downloads."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from predict import AGENT_DIR, load_context, store_root
from research import CRYPTO, EQUITY, features

ASSETS = CRYPTO + EQUITY
HALF_LIVES = [30, 60, 180, 360, 720, 1440, 2880, 10080]


def feature_vector(feature, asset):
    reference = max(feature['av'][360], 1e-8)
    values = [np.log(max(feature[estimator][half_life], 1e-9) / reference)
              for estimator in ['av', 'rv'] for half_life in HALF_LIVES]
    values += [np.log(reference), *[float(asset == name) for name in ASSETS]]
    return np.asarray(values, dtype=np.float32)


def make_asset(asset):
    first = sorted(store_root(asset).glob('date=*.parquet'))[0].stem.removeprefix('date=')
    start = pd.Timestamp(first, tz='UTC')
    end = pd.Timestamp('2026-07-26', tz='UTC')
    prices = load_context(asset, start, end)
    observed = prices.close.dropna()
    returns = np.diff(np.log(observed.to_numpy(dtype=float)))
    gaps = np.diff(observed.index.as_unit('ns').asi8) / 60e9
    if np.any(gaps < 1):
        raise ValueError('Minute-grid timestamps have invalid elapsed-time units')
    normalized = pd.Series(np.abs(returns) / np.sqrt(gaps), index=observed.index[1:]).reindex(prices.index)
    targets = {}
    for horizon in [60, 1440]:
        targets[horizon] = normalized.shift(-1).iloc[::-1].rolling(horizon, min_periods=max(10, horizon//10)).mean().iloc[::-1] * np.sqrt(np.pi / 2)
    rows, labels, stamps = [], [], []
    for timestamp in pd.date_range(start + pd.Timedelta(days=7), end - pd.Timedelta(days=1), freq='3h'):
        context = prices.loc[(prices.index > timestamp - pd.Timedelta(days=7)) & (prices.index <= timestamp)]
        if context.close.notna().sum() < 50:
            continue
        target = [targets[horizon].loc[timestamp] for horizon in [60,1440]]
        if not np.isfinite(target).all() or min(target) <= 0:
            continue
        feature = features(context, timestamp, 60)
        row = feature_vector(feature, asset)
        label = np.asarray(target) / [max(feature['av'][60], 1e-8), max(feature['av'][360], 1e-8)]
        if not np.isfinite(row).all() or not np.isfinite(label).all():
            continue
        rows.append(row)
        labels.append(label)
        stamps.append(timestamp.value)
    print(asset, len(rows), flush=True)
    return np.asarray(rows), np.asarray(labels), np.asarray(stamps)


def train(dataset, output, seed=20260909):
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.35)
    data = np.load(dataset)
    inputs, targets, timestamps = data['inputs'], np.clip(data['targets'], .02, 20), data['timestamps']
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC').value
    validation = ~training
    center = inputs[training].mean(axis=0)
    spread = np.maximum(inputs[training].std(axis=0), 0.1)
    standardized = np.clip((inputs-center)/spread, -8, 8)
    design = np.column_stack([np.ones(len(inputs)), standardized])
    penalty = np.eye(design.shape[1]) * 100
    penalty[0,0] = 1
    ridge = np.linalg.solve(design[training].T @ design[training] + penalty,
                            design[training].T @ (targets[training]-1))
    prediction = np.clip(1 + design @ ridge, 0.2, 4)
    metric = lambda pred, mask: np.mean(targets[mask]/pred[mask] + np.log(pred[mask]), axis=0).tolist()
    metrics = {'seed': seed, 'training_rows': int(training.sum()), 'validation_rows': int(validation.sum()),
               'baseline_validation_loss': metric(np.ones_like(prediction), validation),
               'ridge_validation_loss': metric(prediction, validation), 'device': device,
               'fitting_targets_end_before': '2026-07-26', 'validation_start': '2026-07-01'}
    inputs_tensor = torch.tensor(standardized, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    network = nn.Sequential(nn.Linear(inputs.shape[1],48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=.02)
    indices = torch.tensor(np.flatnonzero(training), device=device)
    valid_indices = torch.tensor(np.flatnonzero(validation), device=device)
    best_loss, best_state, best_epoch = np.inf, None, None
    for epoch in range(150):
        order = indices[torch.randperm(len(indices), device=device)]
        for batch in order.split(1024):
            output_log = network(inputs_tensor[batch]).clamp(-2,2)
            loss = (target_tensor[batch] * torch.exp(-output_log) + output_log + .03 * output_log.square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logs = network(inputs_tensor[valid_indices]).clamp(-2,2)
            loss = (target_tensor[valid_indices] * torch.exp(-logs) + logs).mean().item()
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_state = {key: value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
        if epoch % 25 == 0:
            print('epoch',epoch,'validation_loss',loss,'best',best_loss,flush=True)
    metrics.update({'neural_validation_loss':best_loss,'neural_best_epoch':best_epoch})
    np.savez(output,center=center,spread=spread,ridge=ridge,**best_state)
    output.with_suffix('.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    dataset=AGENT_DIR/'results'/'volatility_training.npz'
    if not dataset.exists():
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows=list(executor.map(make_asset,ASSETS))
        np.savez(dataset,inputs=np.concatenate([row[0] for row in rows]),targets=np.concatenate([row[1] for row in rows]),timestamps=np.concatenate([row[2] for row in rows]))
    output=AGENT_DIR/'volatility_weights.npz'
    train(dataset,output)


if __name__=='__main__':
    main()
