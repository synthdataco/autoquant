"""Chronological selection of frozen-encoder lead-shape corrections."""

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_embedding


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    torch.set_num_threads(2)
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
    timestamps = data['timestamps'][selected][valid]
    targets = np.clip(leads['targets'][positions[valid]], .02, 20)
    weights = dict(np.load(AGENT_DIR/'lead_weights.npz'))
    static = np.clip((inputs-weights['center'])/weights['spread'], -8, 8)
    hidden = static.copy()
    for layer in [0, 2]:
        hidden = hidden @ weights[f'{layer}.weight'].T + weights[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T + weights['4.bias'], -2, 2)
    embedded = np.stack([temporal_embedding(sequence, values, 'temporal_exact')
                         for sequence, values in zip(sequences['sequences'][valid], inputs)])
    design = np.column_stack([embedded, static])
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC').value
    center = design[training].mean(axis=0)
    spread = np.maximum(design[training].std(axis=0), .1)
    design = np.column_stack([np.ones(len(design)), np.clip((design-center)/spread, -8, 8)])
    duration = np.diff(weights['edges'])/weights['edges'][-1]
    training_design = torch.tensor(design[training], dtype=torch.float64)
    training_baseline = torch.tensor(baseline[training], dtype=torch.float64)
    training_targets = torch.tensor(targets[training], dtype=torch.float64)
    durations = torch.tensor(duration, dtype=torch.float64)

    def validation(coefficients):
        correction = np.clip(design[~training] @ coefficients, -.5, .5)
        raw = baseline[~training]+correction
        normalization = np.log((np.exp(raw)*duration).sum(axis=1)/(np.exp(baseline[~training])*duration).sum(axis=1))
        prediction = raw-normalization[:, None]
        return float(np.mean(((targets[~training]*np.exp(-prediction)+prediction)*duration).sum(axis=1)))

    best_coefficients = np.zeros((design.shape[1], 5))
    initial = best = validation(best_coefficients)
    best_penalty = None
    diagnostics = {}
    for penalty in [.01, .1, 1.]:
        def objective(flattened):
            coefficients = torch.tensor(flattened.reshape(design.shape[1], 5), requires_grad=True)
            correction = (training_design @ coefficients).clamp(-.5, .5)
            raw = training_baseline+correction
            normalization = torch.log((raw.exp()*durations).sum(axis=1)/(training_baseline.exp()*durations).sum(axis=1))
            prediction = raw-normalization[:, None]
            loss = ((training_targets*torch.exp(-prediction)+prediction)*durations).sum(axis=1).mean()
            loss = loss + penalty*coefficients.square().sum()
            loss.backward()
            return float(loss.detach()), coefficients.grad.numpy().ravel()

        fit = minimize(objective, best_coefficients.ravel()*0, method='L-BFGS-B', jac=True,
                       options={'maxiter':150, 'ftol':1e-10})
        coefficients = fit.x.reshape(design.shape[1], 5)
        loss = validation(coefficients)
        diagnostics[str(penalty)] = {'loss':loss, 'success':bool(fit.success), 'iterations':int(fit.nit)}
        print(penalty, diagnostics[str(penalty)], flush=True)
        if loss < best:
            best, best_penalty, best_coefficients = loss, penalty, coefficients
    np.savez(AGENT_DIR/'lead_temporal_weights.npz', coefficients=best_coefficients,
             center=center, spread=spread, static_center=weights['center'], static_spread=weights['spread'], duration=duration)
    metrics = {'initial_validation_loss':initial, 'best_validation_loss':best, 'penalty':best_penalty,
               'diagnostics':diagnostics, 'target_cutoff':'2026-07-26', 'validation_start':'2026-07-01',
               'training_rows':int(training.sum()), 'validation_rows':int((~training).sum())}
    (AGENT_DIR/'lead_temporal_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
