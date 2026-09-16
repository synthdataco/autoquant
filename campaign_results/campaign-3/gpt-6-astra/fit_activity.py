"""Chronological quote-activity ablation around the numerical daily stack."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from activity import activity_features
from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from temporal import temporal_embedding, temporal_weights
from tree_vol import tree_design, tree_residual_batch
from train_vol import ASSETS


def make_asset(asset):
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    selected = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')['indices']
    positions = np.flatnonzero(data['inputs'][selected, -13:].argmax(axis=1) == ASSETS.index(asset))
    indices = selected[positions]
    timestamps = pd.to_datetime(data['timestamps'][indices], utc=True)
    prices = load_context(asset, timestamps.min()-pd.Timedelta(days=7), timestamps.max())
    values = [activity_features(prices.loc[timestamp-pd.Timedelta(days=7):timestamp], timestamp, np.exp(data['inputs'][index, 16]))
              for index, timestamp in zip(indices, timestamps)]
    print(asset, len(values), 'activity contexts', flush=True)
    return positions, np.stack(values)


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    threadpool_limits(2)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    sequences = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')
    selected = sequences['indices']
    inputs = data['inputs'][selected]
    cache = AGENT_DIR/'results'/'activity_training.npz'
    if not cache.exists():
        with ProcessPoolExecutor(max_workers=2) as executor:
            batches = list(executor.map(make_asset, ASSETS))
        positions = np.concatenate([batch[0] for batch in batches])
        activity = np.concatenate([batch[1] for batch in batches])[np.argsort(positions)]
        np.savez(cache, indices=selected, activity=activity)
    stored = np.load(cache)
    if not np.array_equal(stored['indices'], selected):
        raise ValueError('Activity cache alignment mismatch')
    previous = dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    hidden = np.clip((inputs-previous['center'])/previous['spread'], -8, 8)
    for layer in [0, 2]:
        hidden = hidden @ previous[f'{layer}.weight'].T+previous[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ previous['4.weight'].T+previous['4.bias'], -2, 2)
    baseline[:, 0] *= .5
    embedded = np.stack([temporal_embedding(sequence, values, 'temporal_exact')
                         for sequence, values in zip(sequences['sequences'], inputs)])
    temporal = temporal_weights('temporal_exact')
    baseline += np.clip(embedded @ temporal['head.2.weight'].T+temporal['head.2.bias'], -.8, .8)
    timestamps = pd.to_datetime(data['timestamps'][selected], utc=True)
    control = tree_design(inputs, sequences['sequences'], timestamps, baseline)['temporal']
    control = np.column_stack([control, embedded])
    baseline += tree_residual_batch(control, 'tree_student')
    designs = {'control':control, 'activity':np.column_stack([control, stored['activity']])}
    targets = np.clip(data['targets'][selected, 1], .02, 20)
    ratios = targets/np.exp(baseline[:, 1])
    training = timestamps < pd.Timestamp('2026-07-01', tz='UTC')
    initial = best = float(np.mean(ratios[~training]+baseline[~training, 1]))
    best_kind, best_stage = None, 0
    exported = {'base':np.zeros(2), 'stages':np.array(0)}
    diagnostics = {}
    for kind, design in designs.items():
        model = HistGradientBoostingRegressor(loss='poisson', max_iter=200, learning_rate=.03,
                                             max_leaf_nodes=7, min_samples_leaf=150, l2_regularization=10.,
                                             early_stopping=False, random_state=95)
        model.fit(design[training], ratios[training])
        for stage, prediction in enumerate(model.staged_predict(design[~training]), 1):
            if stage not in [50, 100, 200]:
                continue
            correction = np.clip(np.log(prediction), -.5, .5)
            loss = float(np.mean(ratios[~training]*np.exp(-correction)+baseline[~training, 1]+correction))
            diagnostics[f'{kind}/{stage}'] = loss
            print(kind, stage, loss, flush=True)
            if loss < best:
                best, best_kind, best_stage = loss, kind, stage
                exported = {'base':np.array([0., model._baseline_prediction.item()]), 'stages':np.array(stage)}
                for number in range(stage):
                    nodes = model._predictors[number][0].nodes
                    exported[f'1_{number}'] = nodes
                    zero = nodes.copy()
                    zero['value'] = 0.
                    exported[f'0_{number}'] = zero
    control_best = min([initial]+[loss for key, loss in diagnostics.items() if key.startswith('control/')])
    np.savez(AGENT_DIR/'activity_weights.npz', **exported)
    metrics = {'initial_validation_loss':initial, 'best_validation_loss':best, 'kind':best_kind, 'stages':best_stage,
               'best_control_loss':control_best, 'passes_activity_gate':bool(best_kind == 'activity' and best < control_best-.0001),
               'diagnostics':diagnostics, 'target_cutoff':'2026-07-26', 'validation_start':'2026-07-01',
               'training_rows':int(training.sum()), 'validation_rows':int((~training).sum())}
    (AGENT_DIR/'activity_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
