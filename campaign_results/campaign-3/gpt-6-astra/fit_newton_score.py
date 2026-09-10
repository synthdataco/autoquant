"""Finite-difference calibration, with every scored fan using predict.score_paths."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from newton_score import newton_design, newton_multiplier
from predict import AGENT_DIR, ensure_workspace_cwd, load_context, score_paths
from research import CRYPTO, EQUITY, features, generate_variant
from synth.validator.moving_average import ASSET_COEFFICIENTS
from synth_lib.backtester.preparation import _slice_real_prices


def build_asset(task):
    asset, verify = task
    ensure_workspace_cwd()
    checkpoint = AGENT_DIR/'results'/'newton_score.parts'/f'{asset}.npz'
    if checkpoint.exists() and not verify:
        return dict(np.load(checkpoint))
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    prior = np.load(AGENT_DIR/'results'/'training_score_curves.npz')
    indices = prior['indices']
    indices = indices[data['inputs'][indices, -13:].argmax(axis=1) == (CRYPTO+EQUITY).index(asset)]
    if verify:
        indices = indices[data['timestamps'][indices] >= pd.Timestamp('2026-07-01', tz='UTC').value]
    timestamps = pd.to_datetime(data['timestamps'][indices], utc=True)
    cutoff = pd.Timestamp('2026-07-26', tz='UTC')
    prices = load_context(asset, timestamps.min()-pd.Timedelta(days=7), cutoff)
    records = []
    for index, timestamp in zip(indices, timestamps):
        if timestamp+pd.Timedelta(days=1) > cutoff:
            continue
        context = prices.loc[(prices.index > timestamp-pd.Timedelta(days=7)) & (prices.index <= timestamp)]
        feature = features(context, timestamp, 300)
        realized = _slice_real_prices(prices[['close']], timestamp, 86400, 300)
        if not np.isfinite(realized[0]) or not np.isfinite(realized[-1]):
            continue
        paths = generate_variant(feature, asset, timestamp, 300, 86400, 1000, 'stacklead:1')
        logs = np.log(paths/feature['last'])
        multipliers = [1., newton_multiplier(feature, asset, timestamp)] if verify else [.92, 1., 1.08]
        scores = [score_paths(feature['last']*np.exp(logs*multiplier), realized, 300, asset, 86400)[0]
                  for multiplier in multipliers]
        if np.isfinite(scores).all() and min(scores) > 0:
            records.append((int(index), newton_design(feature, asset, timestamp), scores))
    result = {'indices':np.array([row[0] for row in records]), 'design':np.stack([row[1] for row in records]),
              'scores':np.array([row[2] for row in records])}
    if not verify:
        checkpoint.parent.mkdir(exist_ok=True)
        temporary = checkpoint.with_suffix('.tmp.npz')
        np.savez(temporary, **result)
        temporary.replace(checkpoint)
    print(asset, len(records), 'verification' if verify else 'curves', flush=True)
    return result


def weights_for(assets, mask, baseline, normalizers=None):
    weights = np.zeros(len(assets))
    scales = []
    for group_index, group in enumerate([CRYPTO, EQUITY]):
        selected_group = np.isin(assets, group)
        for asset in group:
            selected = (assets == asset) & mask
            if selected.any():
                weights[selected] = ASSET_COEFFICIENTS[asset]/selected.sum()
        total = weights[selected_group].sum()
        weights[selected_group] /= max(total, 1e-12)
        scale = float(weights[selected_group] @ baseline[selected_group]) if normalizers is None else normalizers[group_index]
        weights[selected_group] /= max(scale, 1e-8)
        scales.append(scale)
    return weights, scales


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(), flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        batches = list(executor.map(build_asset, [(asset, False) for asset in CRYPTO+EQUITY]))
    stored = {key:np.concatenate([batch[key] for batch in batches]) for key in ['indices', 'design', 'scores']}
    np.savez(AGENT_DIR/'results'/'newton_score_training.npz', **stored)
    original = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    assets = np.asarray(CRYPTO+EQUITY)[original['inputs'][stored['indices'], -13:].argmax(axis=1)]
    training = original['timestamps'][stored['indices']] < pd.Timestamp('2026-07-01', tz='UTC').value
    center = stored['design'][training].mean(axis=0)
    spread = np.maximum(stored['design'][training].std(axis=0), .1)
    design = np.column_stack([np.ones(len(assets)), np.clip((stored['design']-center)/spread, -8, 8)])
    scores = stored['scores']
    gradient = (scores[:, 2]-scores[:, 0])/.16
    curvature = (scores[:, 2]-2*scores[:, 1]+scores[:, 0])/.08**2
    floor = max(float(np.median(curvature[training]))*.001, 1e-6)
    curvature = np.maximum(curvature, floor)
    train_weights, normalizers = weights_for(assets, training, scores[:, 1])
    validation_weights, _ = weights_for(assets, ~training, scores[:, 1], normalizers)
    matrix = design.T @ ((train_weights*curvature)[:, None]*design)
    direction = -(design.T @ (train_weights*gradient))
    coefficients = np.zeros(design.shape[1])
    best, best_penalty = 0., None
    diagnostics = {}
    for penalty in [.001, .01, .1, 1.]:
        candidate = np.linalg.solve(matrix+penalty*np.eye(design.shape[1]), direction)
        change = np.clip(design @ candidate, -.08, .08)
        surrogate = float(validation_weights @ (gradient*change+.5*curvature*change**2))
        diagnostics[str(penalty)] = surrogate
        print(penalty, 'surrogate improvement', surrogate, flush=True)
        if surrogate < best:
            best, best_penalty, coefficients = surrogate, penalty, candidate
    np.savez(AGENT_DIR/'newton_score_weights.npz', center=center, spread=spread, coefficients=coefficients)
    metrics = {'surrogate_change':best, 'penalty':best_penalty, 'diagnostics':diagnostics,
               'curvature_floor':floor, 'training_rows':int(training.sum()), 'validation_rows':int((~training).sum()),
               'target_cutoff':'2026-07-26', 'validation_start':'2026-07-01', 'paths':1000,
               'constant_competition_normalizers':normalizers, 'verification_ratios':{}}
    if best_penalty is not None:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            verified = list(executor.map(build_asset, [(asset, True) for asset in CRYPTO+EQUITY]))
        comparisons = np.concatenate([batch['scores'] for batch in verified])
        verified_indices = np.concatenate([batch['indices'] for batch in verified])
        verified_assets = np.asarray(CRYPTO+EQUITY)[original['inputs'][verified_indices, -13:].argmax(axis=1)]
        weights, _ = weights_for(verified_assets, np.ones(len(verified_assets), dtype=bool), comparisons[:, 0], normalizers)
        for name, group in [('crypto', CRYPTO), ('com_equ', EQUITY)]:
            selected = np.isin(verified_assets, group)
            metrics['verification_ratios'][name] = float((weights[selected] @ comparisons[selected, 1])/(weights[selected] @ comparisons[selected, 0]))
        metrics['passes_verification'] = all(value < 1 for value in metrics['verification_ratios'].values())
        np.savez(AGENT_DIR/'results'/'newton_score_verification.npz', indices=verified_indices, scores=comparisons)
    else:
        metrics['passes_verification'] = False
    (AGENT_DIR/'newton_score_weights.json').write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
