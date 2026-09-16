"""Controlled chronological residual fitting for fast hourly scale features."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from fast_scale import fast_features
from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from research import CRYPTO
from temporal import temporal_embedding, temporal_weights
from tree_vol import tree_design


def make_asset(asset):
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    selected = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')['indices']
    positions = np.flatnonzero(data['inputs'][selected,-13:].argmax(axis=1)==CRYPTO.index(asset))
    indices = selected[positions]
    timestamps = pd.to_datetime(data['timestamps'][indices],utc=True)
    prices = load_context(asset,timestamps.min()-pd.Timedelta(hours=1),timestamps.max())
    values = [fast_features(prices.loc[timestamp-pd.Timedelta(hours=1):timestamp],timestamp,
                            np.exp(data['inputs'][index,16]+data['inputs'][index,1]))
              for timestamp,index in zip(timestamps,indices)]
    print(asset,len(values),'fast contexts',flush=True)
    return positions,np.stack(values)


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    threadpool_limits(1)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    sequences = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')
    cache = AGENT_DIR/'results'/'fast_training.npz'
    if not cache.exists():
        with ProcessPoolExecutor(max_workers=2) as executor:
            batches = list(executor.map(make_asset,CRYPTO))
        positions = np.concatenate([batch[0] for batch in batches])
        order = np.argsort(positions)
        np.savez(cache,positions=positions[order],features=np.concatenate([batch[1] for batch in batches])[order],
                 indices=sequences['indices'][positions[order]])
    stored = np.load(cache)
    positions = stored['positions']
    selected = sequences['indices'][positions]
    assert np.array_equal(selected,stored['indices'])
    inputs = data['inputs'][selected]
    sequence_values = sequences['sequences'][positions]
    previous = dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    hidden = np.clip((inputs-previous['center'])/previous['spread'],-8,8)
    for layer in [0,2]:
        hidden = hidden @ previous[f'{layer}.weight'].T+previous[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ previous['4.weight'].T+previous['4.bias'],-2,2)
    baseline[:,0] *= .5
    embedded = np.stack([temporal_embedding(sequence,values,'temporal_exact')
                         for sequence,values in zip(sequence_values,inputs)])
    temporal = temporal_weights('temporal_exact')
    baseline += np.clip(embedded @ temporal['head.2.weight'].T+temporal['head.2.bias'],-.8,.8)
    timestamps = pd.to_datetime(data['timestamps'][selected],utc=True)
    control = np.column_stack([tree_design(inputs,sequence_values,timestamps,baseline)['temporal'],embedded])
    designs = {'control':control,'fast':np.column_stack([control,stored['features']])}
    ratios = np.clip(data['targets'][selected,0],.02,20)/np.exp(baseline[:,0])
    training = timestamps<pd.Timestamp('2026-07-01',tz='UTC')
    initial = float(np.mean(ratios[~training]+baseline[~training,0]))
    diagnostics = {}
    selected_fits = {}
    for kind,design in designs.items():
        best,best_stage = initial,0
        exported = {'base':np.zeros(2),'stages':np.array(0)}
        model = HistGradientBoostingRegressor(loss='poisson',max_iter=200,learning_rate=.03,
                                             max_leaf_nodes=7,min_samples_leaf=150,l2_regularization=10.,
                                             early_stopping=False,random_state=95)
        model.fit(design[training],ratios[training])
        for stage,prediction in enumerate(model.staged_predict(design[~training]),1):
            if stage not in [50,100,200]:
                continue
            correction = np.clip(np.log(prediction),-.25,.25)
            loss = float(np.mean(ratios[~training]*np.exp(-correction)+baseline[~training,0]+correction))
            diagnostics[f'{kind}/{stage}'] = loss
            print(kind,stage,loss,flush=True)
            if loss<best:
                best,best_stage = loss,stage
                exported = {'base':np.array([model._baseline_prediction.item(),0.]),'stages':np.array(stage)}
                for number in range(stage):
                    nodes = model._predictors[number][0].nodes
                    exported[f'0_{number}'] = nodes
                    zero = nodes.copy()
                    zero['value'] = 0.
                    exported[f'1_{number}'] = zero
        np.savez(AGENT_DIR/f'fast_{kind}_weights.npz',**exported)
        selected_fits[kind] = {'loss':best,'stages':best_stage}
    metrics = {'initial_validation_loss':initial,'selected':selected_fits,'diagnostics':diagnostics,
               'passed_gate':bool(selected_fits['fast']['loss']<selected_fits['control']['loss']-.0005),
               'training_rows':int(training.sum()),'validation_rows':int((~training).sum()),
               'target_cutoff':'2026-07-26','validation_start':'2026-07-01'}
    (AGENT_DIR/'fast_scale_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


if __name__=='__main__':
    main()
