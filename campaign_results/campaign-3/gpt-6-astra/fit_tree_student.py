"""Distill only our own daily teacher into an offline numerical student."""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_embedding, temporal_weights
from tree_vol import tree_design, tree_residual, tree_residual_batch


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    threadpool_limits(2)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    sequences = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')
    selected = sequences['indices']
    inputs = data['inputs'][selected]
    weights = dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    hidden = np.clip((inputs-weights['center'])/weights['spread'], -8, 8)
    for layer in [0, 2]:
        hidden = hidden @ weights[f'{layer}.weight'].T+weights[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T+weights['4.bias'], -2, 2)
    baseline[:, 0] *= .5
    embedded = np.stack([temporal_embedding(sequence, values, 'temporal_exact')
                         for sequence, values in zip(sequences['sequences'], inputs)])
    previous = temporal_weights('temporal_exact')
    baseline += np.clip(embedded @ previous['head.2.weight'].T+previous['head.2.bias'], -.8, .8)
    forecasts = np.load(AGENT_DIR/'results'/'chronos_training.npz')
    if not np.array_equal(forecasts['indices'], selected):
        raise ValueError('Teacher forecast alignment mismatch')
    timestamps = pd.to_datetime(data['timestamps'][selected], utc=True)
    designs = tree_design(inputs, sequences['sequences'], timestamps, baseline, forecasts['forecasts'])
    teacher = tree_residual_batch(designs['pretrained'])
    for index in [0, 29, 121]:
        np.testing.assert_allclose(teacher[index], tree_residual(designs['pretrained'][index]), atol=1e-14)
    design = np.column_stack([designs['temporal'], embedded])
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC')
    models, predictions = [], []
    for head in range(2):
        model = HistGradientBoostingRegressor(loss='squared_error', max_iter=400, learning_rate=.05,
                                             max_leaf_nodes=15, min_samples_leaf=64, l2_regularization=5.,
                                             early_stopping=False, random_state=95)
        model.fit(design[training], teacher[training, head])
        models.append(model)
        predictions.append({stage:prediction for stage, prediction in
                            enumerate(model.staged_predict(design[~training]), 1) if stage in [100, 200, 400]})
    diagnostics = {}
    best, best_stage = np.inf, None
    for stage in [100, 200, 400]:
        predicted = np.column_stack([prediction[stage] for prediction in predictions])
        rmse = np.sqrt(((predicted-teacher[~training])**2).mean(axis=0))
        diagnostics[str(stage)] = rmse.tolist()
        print(stage, rmse, flush=True)
        if rmse[1] < best:
            best, best_stage = float(rmse[1]), stage
    exported = {'base':np.array([model._baseline_prediction.item() for model in models]), 'stages':np.array(best_stage)}
    for head, model in enumerate(models):
        for stage in range(best_stage):
            exported[f'{head}_{stage}'] = model._predictors[stage][0].nodes
    np.savez(AGENT_DIR/'tree_student_weights.npz', **exported)
    relative = best/float(teacher[~training, 1].std())
    metrics = {'daily_rmse':best, 'relative_daily_rmse':relative, 'stages':best_stage,
               'passes_fidelity_gate':bool(best < .02 and relative < .5), 'diagnostics':diagnostics,
               'teacher_label_cutoff':'2026-07-26', 'validation_start':'2026-07-01',
               'training_rows':int(training.sum()), 'validation_rows':int((~training).sum())}
    (AGENT_DIR/'tree_student_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
