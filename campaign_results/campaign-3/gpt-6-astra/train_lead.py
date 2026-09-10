"""Learn future volatility shape from causally split snapshot labels."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from train_vol import ASSETS

EDGES = np.array([0, 60, 180, 360, 720, 1440])


def make_asset(asset):
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    mask = np.argmax(data['inputs'][:, -len(ASSETS):], axis=1) == ASSETS.index(asset)
    inputs = data['inputs'][mask]
    timestamps = pd.to_datetime(data['timestamps'][mask], utc=True)
    prices = load_context(asset, timestamps.min() - pd.Timedelta(minutes=10), pd.Timestamp('2026-07-26', tz='UTC'))
    observed = prices.close.dropna()
    gaps = np.diff(observed.index.as_unit('ns').asi8) / 60e9
    normalized = pd.Series(np.abs(np.diff(np.log(observed.to_numpy()))) / np.sqrt(gaps), index=observed.index[1:]).reindex(prices.index)
    labels = []
    for begin, end in zip(EDGES[:-1], EDGES[1:]):
        width = int(end - begin)
        future = normalized.shift(-int(begin + 1)).iloc[::-1].rolling(width, min_periods=max(10, width // 10)).mean().iloc[::-1]
        labels.append(future.reindex(timestamps).to_numpy() * np.sqrt(np.pi / 2))
    targets = np.stack(labels, axis=1) / np.exp(inputs[:, 16:17])
    valid = np.isfinite(targets).all(axis=1) & (targets > 0).all(axis=1)
    print(asset, int(valid.sum()), flush=True)
    return inputs[valid], targets[valid], timestamps.as_unit('ns').asi8[valid]


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    dataset = AGENT_DIR / 'results' / 'lead_training.npz'
    if not dataset.exists():
        with ProcessPoolExecutor(max_workers=4) as executor:
            batches = list(executor.map(make_asset, ASSETS))
        np.savez(dataset, inputs=np.concatenate([batch[0] for batch in batches]),
                 targets=np.concatenate([batch[1] for batch in batches]),
                 timestamps=np.concatenate([batch[2] for batch in batches]))
    data = np.load(dataset)
    inputs, targets, timestamps = data['inputs'], np.clip(data['targets'], .02, 20), data['timestamps']
    previous = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC').value
    torch.set_num_threads(4)
    torch.manual_seed(73)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    standardized = np.clip((inputs - previous['center']) / previous['spread'], -8, 8)
    values = torch.tensor(standardized, dtype=torch.float32, device=device)
    labels = torch.tensor(targets, dtype=torch.float32, device=device)
    weights = torch.tensor(np.diff(EDGES) / EDGES[-1], dtype=torch.float32, device=device)
    network = nn.Sequential(nn.Linear(30, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 5)).to(device)
    with torch.no_grad():
        for layer in [0, 2]:
            network[layer].weight.copy_(torch.tensor(previous[f'{layer}.weight'], device=device))
            network[layer].bias.copy_(torch.tensor(previous[f'{layer}.bias'], device=device))
        network[4].weight.copy_(torch.tensor(np.repeat(previous['4.weight'][1:2], 5, axis=0), device=device))
        network[4].bias.copy_(torch.tensor(np.repeat(previous['4.bias'][1], 5), device=device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0005, weight_decay=.04)
    indices = torch.tensor(np.flatnonzero(training), device=device)
    validation = torch.tensor(np.flatnonzero(~training), device=device)

    def loss_for(selected, regularize=False):
        logs = network(values[selected]).clamp(-2, 2)
        loss = ((labels[selected] * torch.exp(-logs) + logs) * weights).sum(axis=1).mean()
        if regularize:
            loss = loss + .02 * logs.square().mean() + .02 * torch.diff(logs, dim=1).square().mean()
        return loss

    initial = float(loss_for(validation).detach().cpu())
    best, best_epoch = initial, -1
    best_state = {key: value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
    for epoch in range(180):
        for batch in indices[torch.randperm(len(indices), device=device)].split(1024):
            optimizer.zero_grad()
            loss_for(batch, True).backward()
            optimizer.step()
        with torch.no_grad():
            loss = float(loss_for(validation).cpu())
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {key: value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
        if epoch % 30 == 0:
            print(epoch, loss, best, flush=True)
    np.savez(AGENT_DIR / 'lead_weights.npz', center=previous['center'], spread=previous['spread'], edges=EDGES, **best_state)
    metrics = {'initial_validation_loss': initial, 'best_validation_loss': best, 'best_epoch': best_epoch,
               'training_rows': int(training.sum()), 'validation_rows': int((~training).sum()),
               'target_cutoff': '2026-07-26', 'validation_start': '2026-07-01', 'edges_minutes': EDGES.tolist()}
    (AGENT_DIR / 'lead_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
