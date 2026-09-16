"""Estimate positive latent-scale pools from past forecasting errors."""

import json
import math

import numpy as np
import pandas as pd

from predict import AGENT_DIR, ensure_workspace_cwd

ensure_workspace_cwd()
print((AGENT_DIR / 'BUDGET.md').read_text(),flush=True)
data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
weights = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
inputs = data['inputs']
hidden = np.clip((inputs-weights['center'])/weights['spread'],-8,8)
for layer in [0,2]:
    hidden = hidden @ weights[f'{layer}.weight'].T+weights[f'{layer}.bias']
    hidden = hidden/(1+np.exp(-hidden))
logs = np.clip(hidden @ weights['4.weight'].T+weights['4.bias'],-2,2)
logs[:,0] *= .5
residual = np.clip(data['targets']/np.exp(logs),.05,10)
training = data['timestamps'] < pd.Timestamp('2026-07-01',tz='UTC').value
group = (np.argmax(inputs[:,-13:],axis=1)>=5).astype(int)
regime = np.digitize(inputs[:,1],[-.2,.2])
probabilities = (np.arange(101)+.5)/101
expected_scale = math.sqrt(3/2)*math.gamma(2)/math.gamma(2.5)
pooled = np.ones((2,2,101))*expected_scale
conditional = np.ones((2,2,3,101))*expected_scale
counts = {}
for head in range(2):
    for category in range(2):
        mask = training & (group==category)
        quantiles = np.quantile(residual[mask,head],probabilities)
        pooled[head,category] = quantiles/quantiles.mean()*expected_scale
        for state in range(3):
            selected = mask & (regime==state)
            quantiles = np.quantile(residual[selected,head],probabilities) if selected.sum()>=100 else pooled[head,category]
            conditional[head,category,state] = quantiles/quantiles.mean()*expected_scale
            counts[f'{head}/{category}/{state}'] = int(selected.sum())
np.savez(AGENT_DIR / 'uncertainty_weights.npz',pooled=pooled,conditional=conditional)
metrics = {'fitting_targets_end_before':'2026-07-02','training_start_times_before':'2026-07-01',
           'expected_path_scale':expected_scale,'counts':counts,'quantiles':101,
           'interpretation':'Latent future-volatility forecast-error pools; not raw-return bootstrapping.'}
(AGENT_DIR / 'uncertainty_weights.json').write_text(json.dumps(metrics,indent=2))
print(metrics,flush=True)
