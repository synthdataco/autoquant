"""Paired fixed-grid versus randomized-clock training, with identical label days."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from newton_score import newton_design
from periodic import periodic_multiplier
from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from research import CRYPTO, EQUITY, features
from tree_vol import tree_residual


def make_asset(task):
    asset, fingerprint = task
    checkpoint = AGENT_DIR/'results'/'phase_training.parts'/f'{asset}_{fingerprint}.npz'
    if checkpoint.exists():
        return dict(np.load(checkpoint))
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    selected = data['inputs'][:, -13:].argmax(axis=1) == (CRYPTO+EQUITY).index(asset)
    anchors = pd.to_datetime(data['timestamps'][selected], utc=True)
    cutoff = pd.Timestamp('2026-07-26', tz='UTC')
    prices = load_context(asset, anchors.min()-pd.Timedelta(days=7), cutoff)
    observed = prices.close.dropna()
    gaps = np.diff(observed.index.as_unit('ns').asi8)/60e9
    absolute = pd.Series(np.abs(np.diff(np.log(observed.to_numpy())))/np.sqrt(gaps), index=observed.index[1:]).reindex(prices.index)
    future = absolute.shift(-1).iloc[::-1].rolling(1440, min_periods=144).mean().iloc[::-1]*np.sqrt(np.pi/2)
    rows, targets, baselines, timestamps = [], [], [], []
    for anchor in anchors:
        offset = int.from_bytes(hashlib.blake2s(f'527:{asset}:{anchor}'.encode(), digest_size=4).digest(), 'little') % 180
        pair = [anchor, anchor+pd.Timedelta(minutes=offset)]
        if pair[-1]+pd.Timedelta(days=1) > cutoff:
            continue
        pair_rows, pair_targets, pair_baselines = [], [], []
        for timestamp in pair:
            context = prices.loc[(prices.index > timestamp-pd.Timedelta(days=7)) & (prices.index <= timestamp)]
            target = future.loc[timestamp]
            if context.close.notna().sum() < 50 or not np.isfinite(target) or target <= 0:
                break
            feature = features(context, timestamp, 300)
            design = newton_design(feature, asset, timestamp)
            baseline = design[31]+tree_residual(design, 'tree_student')[1]
            if asset not in CRYPTO:
                baseline += .5*np.log(periodic_multiplier((CRYPTO+EQUITY).index(asset), timestamp, 86400))
            pair_rows.append(design)
            pair_targets.append(float(np.clip(target/max(feature['av'][360], 1e-8), .02, 20)))
            pair_baselines.append(baseline)
        if len(pair_rows) == 2:
            rows.append(pair_rows)
            targets.append(pair_targets)
            baselines.append(pair_baselines)
            timestamps.append(anchor.value)
    output = {'design':np.array(rows), 'targets':np.array(targets), 'baseline':np.array(baselines), 'timestamps':np.array(timestamps)}
    checkpoint.parent.mkdir(exist_ok=True)
    temporary = checkpoint.with_suffix('.tmp.npz')
    np.savez(temporary, **output)
    temporary.replace(checkpoint)
    print(asset, len(rows), 'paired anchors', flush=True)
    return output


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    threadpool_limits(2)
    digest = hashlib.sha256()
    for name in ['volatility', 'temporal_exact', 'tree_student', 'periodic']:
        digest.update((AGENT_DIR/f'{name}_weights.npz').read_bytes())
    fingerprint = digest.hexdigest()[:16]
    with ProcessPoolExecutor(max_workers=4) as executor:
        batches = list(executor.map(make_asset, [(asset, fingerprint) for asset in CRYPTO+EQUITY]))
    data = {key:np.concatenate([batch[key] for batch in batches]) for key in ['design', 'targets', 'baseline', 'timestamps']}
    np.savez(AGENT_DIR/'results'/'phase_training.npz', **data)
    training = data['timestamps'] < pd.Timestamp('2026-07-01', tz='UTC').value
    ratios = data['targets']/np.exp(data['baseline'])
    initial = float(np.mean(ratios[~training, 1]+data['baseline'][~training, 1]))
    diagnostics, bests = {}, {'control':initial, 'shifted':initial}
    best_stage = 0
    exports = {name:{'base':np.zeros(2), 'stages':np.array(0)} for name in ['control','shifted']}
    for position, name in enumerate(['control', 'shifted']):
        model = HistGradientBoostingRegressor(loss='poisson', max_iter=200, learning_rate=.03,
                                             max_leaf_nodes=7, min_samples_leaf=150, l2_regularization=10.,
                                             early_stopping=False, random_state=95)
        model.fit(data['design'][training, position], ratios[training, position])
        for stage, prediction in enumerate(model.staged_predict(data['design'][~training, 1]), 1):
            if stage not in [50, 100, 200]:
                continue
            correction = np.clip(np.log(prediction), -.5, .5)
            loss = float(np.mean(ratios[~training, 1]*np.exp(-correction)+data['baseline'][~training, 1]+correction))
            diagnostics[f'{name}/{stage}'] = loss
            print(name, stage, loss, flush=True)
            if loss < bests[name]:
                bests[name] = loss
                if name == 'shifted':
                    best_stage = stage
                exported = {'base':np.array([0., model._baseline_prediction.item()]), 'stages':np.array(stage)}
                for number in range(stage):
                    nodes = model._predictors[number][0].nodes
                    exported[f'1_{number}'] = nodes
                    zero = nodes.copy()
                    zero['value'] = 0.
                    exported[f'0_{number}'] = zero
                exports[name] = exported
    np.savez(AGENT_DIR/'phase_weights.npz', **exports['shifted'])
    np.savez(AGENT_DIR/'phase_control_weights.npz', **exports['control'])
    metrics = {'initial_validation_loss':initial, 'best_control_loss':bests['control'], 'best_shifted_loss':bests['shifted'],
               'stages':best_stage, 'passes_phase_gate':bool(bests['shifted'] < min(initial, bests['control'])-.0005),
               'diagnostics':diagnostics, 'target_cutoff':'2026-07-26', 'validation_start':'2026-07-01',
               'training_pairs':int(training.sum()), 'validation_pairs':int((~training).sum()), 'weight_fingerprint':fingerprint}
    (AGENT_DIR/'phase_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
