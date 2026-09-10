"""Fit a small sequence residual over the frozen static forecaster."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from temporal import context_sequence, hourly_observations, normalize_sequence
from train_vol import ASSETS


def make_asset(task):
    asset, exact = task
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    indices = np.flatnonzero(np.argmax(data['inputs'][:, -len(ASSETS):], axis=1) == ASSETS.index(asset))
    timestamps = pd.to_datetime(data['timestamps'][indices], utc=True)
    prices = load_context(asset, timestamps.min() - pd.Timedelta(days=7), timestamps.max())
    if exact:
        sequences = [context_sequence(prices.loc[timestamp-pd.Timedelta(days=7):timestamp], timestamp, np.exp(data['inputs'][index, 16]))
                     for timestamp, index in zip(timestamps, indices)]
    else:
        hourly = hourly_observations(prices.close)
        sequences = [normalize_sequence(hourly, timestamp, np.exp(data['inputs'][index, 16]))
                     for timestamp, index in zip(timestamps, indices)]
    print(asset, len(sequences), flush=True)
    return indices, np.stack(sequences)


class TemporalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(4, 8, 5, stride=2, padding=2), nn.SiLU(),
                                     nn.Conv1d(8, 16, 5, stride=2, padding=2), nn.SiLU(),
                                     nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(16 * 21 + 30, 48), nn.SiLU(), nn.Linear(48, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequence, static):
        encoded = self.encoder(sequence).flatten(1)
        return self.head(torch.cat([encoded, static], dim=1)).clamp(-.8, .8)


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exact-context', action='store_true')
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--output-name')
    args = parser.parse_args()
    name = 'temporal_exact' if args.exact_context else 'temporal'
    output_name = args.output_name or name
    path = AGENT_DIR / 'results' / f'{name}_training.npz'
    if not path.exists():
        with ProcessPoolExecutor(max_workers=2) as executor:
            batches = list(executor.map(make_asset, [(asset,args.exact_context) for asset in ASSETS]))
        np.savez(path, indices=np.concatenate([batch[0] for batch in batches]), sequences=np.concatenate([batch[1] for batch in batches]))
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    temporal = np.load(path)
    selected = temporal['indices']
    previous = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
    static = np.clip((data['inputs'][selected] - previous['center']) / previous['spread'], -8, 8)
    baseline = static.copy()
    for layer in [0, 2]:
        baseline = baseline @ previous[f'{layer}.weight'].T + previous[f'{layer}.bias']
        baseline = baseline / (1 + np.exp(-baseline))
    baseline = np.clip(baseline @ previous['4.weight'].T + previous['4.bias'], -2, 2)
    training = data['timestamps'][selected] < pd.Timestamp('2026-07-01', tz='UTC').value
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    static = torch.tensor(static, dtype=torch.float32, device=device)
    sequences = torch.tensor(temporal['sequences'], dtype=torch.float32, device=device)
    baseline = torch.tensor(baseline, dtype=torch.float32, device=device)
    targets = torch.tensor(np.clip(data['targets'][selected], .02, 20), dtype=torch.float32, device=device)
    indices = torch.tensor(np.flatnonzero(training), device=device)
    validation = torch.tensor(np.flatnonzero(~training), device=device)
    network = TemporalNetwork().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0003, weight_decay=.08)

    def loss_for(batch, penalty=0.):
        residual = network(sequences[batch], static[batch])
        output = (baseline[batch] + residual).clamp(-2, 2)
        return (targets[batch] * torch.exp(-output) + output + penalty * residual.square()).mean()

    initial = float(loss_for(validation).detach().cpu())
    best, best_epoch = initial, -1
    best_state = {key: value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
    for epoch in range(100):
        for batch in indices[torch.randperm(len(indices), device=device)].split(512):
            optimizer.zero_grad()
            loss_for(batch, .12).backward()
            optimizer.step()
        with torch.no_grad():
            loss = float(loss_for(validation).cpu())
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {key: value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
        if epoch % 10 == 0:
            print(epoch, loss, best, flush=True)
    np.savez(AGENT_DIR / f'{output_name}_weights.npz', center=previous['center'], spread=previous['spread'], **best_state)
    metrics = {'initial_validation_loss': initial, 'best_validation_loss': best, 'best_epoch': best_epoch,
               'target_cutoff': '2026-07-26', 'validation_start': '2026-07-01',
               'training_rows': int(training.sum()), 'validation_rows': int((~training).sum()), 'exact_context': args.exact_context,
               'seed': args.seed}
    (AGENT_DIR / f'{output_name}_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
