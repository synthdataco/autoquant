"""Select shallow volatility residuals before examining field outcomes."""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_residual
from tree_vol import tree_design


def main():
    ensure_workspace_cwd()
    threadpool_limits(3)
    data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
    temporal = np.load(AGENT_DIR / 'results' / 'temporal_exact_training.npz')
    selected = temporal['indices']
    inputs = data['inputs'][selected]
    weights = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
    hidden = np.clip((inputs-weights['center'])/weights['spread'], -8, 8)
    for layer in [0, 2]:
        hidden = hidden @ weights[f'{layer}.weight'].T + weights[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T + weights['4.bias'], -2, 2)
    baseline[:, 0] *= .5
    for index, (sequence, static) in enumerate(zip(temporal['sequences'], inputs)):
        baseline[index] += temporal_residual(sequence, static, 'temporal_exact')
    chronos = np.load(AGENT_DIR / 'results' / 'chronos_training.npz')
    if not np.array_equal(chronos['indices'], selected):
        raise ValueError('Pretrained cache alignment mismatch')
    timestamps = pd.to_datetime(data['timestamps'][selected], utc=True)
    designs = tree_design(inputs, temporal['sequences'], timestamps, baseline, chronos['forecasts'])
    targets = np.clip(data['targets'][selected], .02, 20)
    ratios = targets/np.exp(baseline)
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC')
    initial = float(np.mean(ratios[~training]+baseline[~training]))
    best, best_kind, best_stage = initial, None, 0
    exported = {'base': np.zeros(2), 'stages': np.array(0)}
    diagnostics = {}
    for kind, design in designs.items():
        models, predictions = [], []
        for head in range(2):
            model = HistGradientBoostingRegressor(loss='poisson', max_iter=200, learning_rate=.03,
                                                 max_leaf_nodes=7, min_samples_leaf=150,
                                                 l2_regularization=10., early_stopping=False, random_state=95)
            model.fit(design[training], ratios[training, head])
            models.append(model)
            predictions.append({stage: np.log(prediction) for stage, prediction in
                                enumerate(model.staged_predict(design[~training]), 1) if stage in [50, 100, 200]})
        for stage in [50, 100, 200]:
            correction = np.clip(np.column_stack([prediction[stage] for prediction in predictions]), -.5, .5)
            loss = float(np.mean(ratios[~training]*np.exp(-correction)+baseline[~training]+correction))
            diagnostics[f'{kind}/{stage}'] = loss
            print(kind, stage, loss, flush=True)
            if loss < best:
                best, best_kind, best_stage = loss, kind, stage
                exported = {'base': np.array([model._baseline_prediction.item() for model in models]), 'stages': np.array(stage)}
                for head, model in enumerate(models):
                    for number in range(stage):
                        exported[f'{head}_{number}'] = model._predictors[number][0].nodes
    np.savez(AGENT_DIR / 'tree_weights.npz', **exported)
    metrics = {'initial_validation_loss': initial, 'best_validation_loss': best, 'kind': best_kind,
               'stages': best_stage, 'diagnostics': diagnostics, 'label_cutoff': '2026-07-26',
               'training_rows': int(training.sum()), 'validation_rows': int((~training).sum())}
    (AGENT_DIR / 'tree_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
