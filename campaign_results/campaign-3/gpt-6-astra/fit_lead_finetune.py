"""Fit a mean-preserving nonlinear temporal lead correction."""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_embedding
from train_temporal import TemporalNetwork


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(73)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    sequences = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')
    leads = np.load(AGENT_DIR/'results'/'lead_training.npz')
    selected = sequences['indices']
    inputs = data['inputs'][selected]
    keys = pd.MultiIndex.from_arrays([data['timestamps'][selected], inputs[:, -13:].argmax(axis=1)])
    lead_keys = pd.MultiIndex.from_arrays([leads['timestamps'], leads['inputs'][:, -13:].argmax(axis=1)])
    positions = lead_keys.get_indexer(keys)
    valid = positions >= 0
    inputs = inputs[valid]
    targets = np.clip(leads['targets'][positions[valid]], .02, 20)
    weights = dict(np.load(AGENT_DIR/'lead_weights.npz'))
    static = np.clip((inputs-weights['center'])/weights['spread'], -8, 8)
    hidden = static.copy()
    for layer in [0, 2]:
        hidden = hidden @ weights[f'{layer}.weight'].T+weights[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T+weights['4.bias'], -2, 2)
    embedded = np.stack([temporal_embedding(sequence, values, 'temporal_exact')
                         for sequence, values in zip(sequences['sequences'][valid], inputs)])
    linear = dict(np.load(AGENT_DIR/'lead_temporal_weights.npz'))
    design = np.column_stack([embedded, static])
    design = np.column_stack([np.ones(len(design)), np.clip((design-linear['center'])/linear['spread'], -8, 8)])
    raw = baseline+np.clip(design @ linear['coefficients'], -.5, .5)
    duration = np.diff(weights['edges'])/weights['edges'][-1]
    baseline = raw-np.log((np.exp(raw)*duration).sum(axis=1)/(np.exp(baseline)*duration).sum(axis=1))[:, None]
    training = data['timestamps'][selected][valid] < pd.Timestamp('2026-07-01', tz='UTC').value
    network = TemporalNetwork()
    network.head[2] = nn.Linear(48, 5)
    previous = dict(np.load(AGENT_DIR/'temporal_exact_weights.npz'))
    network.load_state_dict({key:torch.tensor(previous[key]) for key in network.state_dict() if not key.startswith('head.2')}, strict=False)
    nn.init.zeros_(network.head[2].weight)
    nn.init.zeros_(network.head[2].bias)
    network = network.to(device)
    sequence_tensor = torch.tensor(sequences['sequences'][valid], dtype=torch.float32, device=device)
    static_tensor = torch.tensor(static, dtype=torch.float32, device=device)
    labels = torch.tensor(targets, dtype=torch.float32, device=device)
    baseline_tensor = torch.tensor(baseline, dtype=torch.float32, device=device)
    duration_tensor = torch.tensor(duration, dtype=torch.float32, device=device)
    indices = torch.tensor(np.flatnonzero(training), device=device)
    validation = torch.tensor(np.flatnonzero(~training), device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0002, weight_decay=.1)

    def loss_for(batch, regularize=False):
        correction = network(sequence_tensor[batch], static_tensor[batch]).clamp(-.5, .5)
        raw = baseline_tensor[batch]+correction
        normalization = torch.log((raw.exp()*duration_tensor).sum(axis=1)/(baseline_tensor[batch].exp()*duration_tensor).sum(axis=1))
        prediction = raw-normalization[:, None]
        loss = ((labels[batch]*torch.exp(-prediction)+prediction)*duration_tensor).sum(axis=1).mean()
        if regularize:
            loss = loss+.05*correction.square().mean()+.02*torch.diff(correction, dim=1).square().mean()
        return loss

    initial = best = float(loss_for(validation).detach().cpu())
    best_epoch = -1
    best_state = {key:value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
    for epoch in range(120):
        for batch in indices[torch.randperm(len(indices), device=device)].split(512):
            optimizer.zero_grad()
            loss_for(batch, True).backward()
            optimizer.step()
        with torch.no_grad():
            loss = float(loss_for(validation).cpu())
        if loss < best:
            best, best_epoch = loss, epoch
            best_state = {key:value.detach().cpu().numpy().copy() for key, value in network.state_dict().items()}
        if epoch % 20 == 0:
            print(epoch, loss, best, flush=True)
    np.savez(AGENT_DIR/'lead_finetune_weights.npz', center=weights['center'], spread=weights['spread'], duration=duration, **best_state)
    metrics = {'initial_validation_loss':initial, 'best_validation_loss':best, 'best_epoch':best_epoch,
               'training_rows':int(training.sum()), 'validation_rows':int((~training).sum()),
               'target_cutoff':'2026-07-26', 'validation_start':'2026-07-01', 'seed':73, 'epochs':120}
    (AGENT_DIR/'lead_finetune_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
