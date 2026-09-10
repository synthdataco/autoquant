"""Chronological evaluation of frozen pretrained volatility features."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from chronos_vol import MODEL_REVISION, calibration_design, forecast_features
from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_residual


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR / 'BUDGET.md').read_text(), flush=True)
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    temporal = np.load(AGENT_DIR / 'results' / 'temporal_exact_training.npz')
    selected = temporal['indices']
    inputs = data['inputs'][selected]
    weights = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
    hidden = np.clip((inputs - weights['center']) / weights['spread'], -8, 8)
    for layer in [0, 2]:
        hidden = hidden @ weights[f'{layer}.weight'].T + weights[f'{layer}.bias']
        hidden = hidden / (1 + np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T + weights['4.bias'], -2, 2)
    baseline[:, 0] *= .5
    for index, (sequence, static) in enumerate(zip(temporal['sequences'], inputs)):
        baseline[index] += temporal_residual(sequence, static, 'temporal_exact')
    cache = AGENT_DIR / 'results' / 'chronos_training.npz'
    if not cache.exists():
        batches = []
        for begin in range(0, len(inputs), 128):
            batches.append(forecast_features(temporal['sequences'][begin:begin+128]))
            if begin % 2048 == 0:
                print('forecasts', begin, len(inputs), flush=True)
        np.savez(cache, forecasts=np.concatenate(batches), indices=selected)
    cached = np.load(cache)
    if not np.array_equal(cached['indices'], selected):
        raise ValueError('Chronos training cache alignment mismatch')
    design = calibration_design(cached['forecasts'], baseline, inputs[:, 1])
    targets = np.clip(data['targets'][selected], .02, 20)
    ratio = targets / np.exp(baseline)
    training = data['timestamps'][selected] < pd.Timestamp('2026-07-01', tz='UTC').value
    initial = float(np.mean(ratio[~training] + baseline[~training]))
    diagnostics = {}
    coefficients_by_kind = {}
    for kind, dimensions in [('intercept', 1), ('pretrained', 4)]:
        best, best_penalty = initial, None
        best_coefficients = np.zeros((2, 4))
        current = design[:, :, :dimensions]
        for penalty in [.001, .01, .1]:
            def objective(flattened):
                coefficients = flattened.reshape(2, dimensions)
                raw = np.einsum('nhd,hd->nh', current[training], coefficients)
                correction = np.clip(raw, -.5, .5)
                scaled = ratio[training] * np.exp(-correction)
                loss = np.mean(scaled + correction) + penalty * np.sum(coefficients**2)
                gradient = np.einsum('nhd,nh->hd', current[training], (1-scaled)*(np.abs(raw)<.5)) / scaled.size
                return float(loss), (gradient + 2*penalty*coefficients).ravel()

            fitted = minimize(objective, np.zeros(2*dimensions), method='L-BFGS-B', jac=True)
            coefficients = fitted.x.reshape(2, dimensions)
            correction = np.clip(np.einsum('nhd,hd->nh', current[~training], coefficients), -.5, .5)
            loss = float(np.mean(ratio[~training]*np.exp(-correction) + baseline[~training] + correction))
            diagnostics[f'{kind}/{penalty}'] = {'loss': loss, 'success': bool(fitted.success)}
            if loss < best:
                best, best_penalty = loss, penalty
                best_coefficients[:, :dimensions] = coefficients
        diagnostics[kind] = {'best_validation_loss': best, 'penalty': best_penalty}
        coefficients_by_kind[kind] = best_coefficients
    np.savez(AGENT_DIR / 'chronos_calibration_weights.npz', **coefficients_by_kind)
    metrics = {'initial_validation_loss': initial, 'diagnostics': diagnostics,
               'model_revision': MODEL_REVISION, 'label_cutoff': '2026-07-26',
               'training_rows': int(training.sum()), 'validation_rows': int((~training).sum())}
    (AGENT_DIR / 'chronos_calibration_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
