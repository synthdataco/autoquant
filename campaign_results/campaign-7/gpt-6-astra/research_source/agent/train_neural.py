"""Bounded GPU training of small chronological volatility sequence models."""
from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import json
import os
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn

from ml_features import ASSETS
from neural_features import NUMERIC_NAMES, SCALES
from run_suite import ROOT


class VolatilityNet(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('numeric_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('numeric_std', torch.as_tensor(std, dtype=torch.float32))
        self.short = nn.GRU(3, 16, batch_first=True)
        self.medium = nn.GRU(3, 16, batch_first=True)
        self.long = nn.GRU(3, 16, batch_first=True)
        self.asset = nn.Embedding(len(ASSETS), 8)
        self.hidden = nn.Sequential(nn.Linear(48+len(NUMERIC_NAMES)+8, 64), nn.SiLU(),
                                    nn.Dropout(.1), nn.Linear(64, 32), nn.SiLU())
        self.output = nn.Linear(32, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, short, medium, long, numeric, asset):
        _, hs = self.short(short)
        _, hm = self.medium(medium)
        _, hl = self.long(long)
        numeric = (numeric-self.numeric_mean)/self.numeric_std
        features = torch.cat([hs[-1], hm[-1], hl[-1], numeric, self.asset(asset)], dim=1)
        return self.output(self.hidden(features)).squeeze(-1)


@lru_cache(maxsize=2)
def data_for(horizon):
    files = sorted((ROOT/'agent/cache').glob(f'sequence_*_{horizon}.npz'))
    expected = 5 if horizon == 3600 else 13
    if len(files) != expected:
        raise RuntimeError('Complete all sequence inputs before fitting')
    keys = ['seq0', 'seq1', 'seq2', 'numeric', 'asset', 'y', 'start_ns', 'end_ns']
    parts = {k: [] for k in keys}
    for path in files:
        with np.load(path) as data:
            for key in keys:
                parts[key].append(data[key])
    return {k: np.concatenate(parts[k], axis=0) for k in keys}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(pack, numeric, train_idx, valid_idx, epochs, seed, batch_size, probe=True):
    set_seed(seed)
    train_np = train_idx.cpu().numpy()
    mean = numeric[train_np].mean(axis=0)
    std = np.maximum(numeric[train_np].std(axis=0), .05)
    model = VolatilityNet(mean, std).cuda()
    with torch.no_grad():
        model.output.bias.fill_(float(pack[-1][train_idx].mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    best, best_epoch, stale = float('inf'), 1, 0
    history = []
    for epoch in range(1, epochs+1):
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            raise RuntimeError('Landing order stopped the neural experiment')
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device='cuda', generator=generator)]
        total = 0.
        for first in range(0, len(order), batch_size):
            batch = order[first:first+batch_size]
            prediction = model(*(x[batch] for x in pack[:-1]))
            loss = nn.functional.mse_loss(prediction, pack[-1][batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(batch)
        train_mse = total/len(order)
        valid_mse = None
        if probe:
            model.eval()
            total = 0.
            with torch.inference_mode():
                for first in range(0, len(valid_idx), 4096):
                    batch = valid_idx[first:first+4096]
                    prediction = model(*(x[batch] for x in pack[:-1]))
                    total += float(nn.functional.mse_loss(prediction, pack[-1][batch], reduction='sum'))
            valid_mse = total/len(valid_idx)
            if valid_mse < best-1e-5:
                best, best_epoch, stale = valid_mse, epoch, 0
            else:
                stale += 1
        history.append({'epoch': epoch, 'train_log_mse': train_mse, 'valid_log_mse': valid_mse})
        if epoch == 1 or epoch % 10 == 0:
            print('probe' if probe else 'refit', epoch, train_mse, valid_mse, flush=True)
        if probe and stale >= 8:
            break
    return model, best_epoch, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--dates', nargs='+', default=['2026-03-01', '2026-07-01', '2026-08-01', '2026-10-01', '2026-11-01', '2026-11-09'])
    args = ap.parse_args()
    os.chdir(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded training run expects the supplied GPU')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed = int(os.environ.get('SYNTH_BENCHMARK_SEED', '20261108'))
    dest = ROOT/'agent/artifacts/neural'
    dest.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'agent/results/neural'
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {'numeric_features': NUMERIC_NAMES, 'scales': SCALES, 'assets': ASSETS,
                'target': 'log forward absolute-return scale / original causal seasonal EWMA',
                'torch_version': torch.__version__, 'seed': seed, 'fits': []}
    for date in args.dates:
        cutoff = pd.Timestamp(date, tz='UTC')
        for horizon in [3600, 86400]:
            if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
                return
            data = data_for(horizon)
            cutoff_ns = cutoff.value
            split_ns = (cutoff-pd.Timedelta(days=14)).value
            mask = (data['end_ns'] < cutoff_ns) & (data['start_ns'] >= (cutoff-pd.Timedelta(days=240)).value)
            selected = {k: v[mask] for k, v in data.items()}
            train = np.flatnonzero(selected['end_ns'] < split_ns)
            valid = np.flatnonzero(selected['start_ns'] >= split_ns)
            if len(train) < 1000 or len(valid) < 100:
                continue
            print('Fit', date, horizon, 'rows', len(selected['y']), 'train', len(train), 'valid', len(valid), flush=True)
            inputs = ['seq0', 'seq1', 'seq2', 'numeric', 'asset', 'y']
            pack = [torch.as_tensor(selected[k], device='cuda') for k in inputs]
            train_idx = torch.as_tensor(train, dtype=torch.long, device='cuda')
            valid_idx = torch.as_tensor(valid, dtype=torch.long, device='cuda')
            probe, best_epoch, best_loss, history = fit(pack, selected['numeric'], train_idx, valid_idx,
                                                       args.epochs, seed, args.batch_size, True)
            del probe
            gc.collect()
            torch.cuda.empty_cache()
            full_idx = torch.arange(len(selected['y']), device='cuda')
            model, _, _, refit_history = fit(pack, selected['numeric'], full_idx, valid_idx,
                                              best_epoch, seed, args.batch_size, False)
            model = model.cpu().eval()
            examples = tuple(torch.as_tensor(selected[k][:1]) for k in inputs[:-1])
            with torch.inference_mode():
                frozen = torch.jit.freeze(torch.jit.trace(model, examples))
                delta = float(torch.max(torch.abs(model(*examples)-frozen(*examples))))
            filename = f'volatility_{horizon}_{date}.pt'
            frozen.save(str(dest/filename))
            manifest['fits'].append({'cutoff': date, 'time_length': horizon, 'file': filename,
                                     'training_rows': len(selected['y']), 'purged_training_rows': len(train),
                                     'validation_rows': len(valid), 'epochs': best_epoch,
                                     'past_validation_log_rmse': float(np.sqrt(best_loss)),
                                     'cpu_export_max_difference': delta})
            (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            (logs/f'{horizon}_{date}.json').write_text(json.dumps({'probe': history, 'refit': refit_history}, indent=2)+'\n')
            print('Saved', filename, 'epochs', best_epoch, 'past log RMSE', np.sqrt(best_loss), 'export difference', delta, flush=True)
            del pack, model, frozen, selected, train_idx, valid_idx, full_idx
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
