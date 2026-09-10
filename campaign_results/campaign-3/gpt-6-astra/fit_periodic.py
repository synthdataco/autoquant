"""Chronological harmonic likelihood fitting around the tuned baseline."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from periodic import periodic_design
from predict import AGENT_DIR, ensure_workspace_cwd

ensure_workspace_cwd()
print((AGENT_DIR / 'BUDGET.md').read_text(),flush=True)
data = np.load(AGENT_DIR / 'results' / 'volatility_training.npz')
previous = dict(np.load(AGENT_DIR / 'volatility_weights.npz'))
inputs = data['inputs']
timestamps = pd.to_datetime(data['timestamps'],utc=True)
training = timestamps < pd.Timestamp('2026-07-01',tz='UTC')
design = periodic_design(timestamps,np.argmax(inputs[:,-13:],axis=1))
hidden = np.clip((inputs-previous['center'])/previous['spread'],-8,8)
for layer in [0,2]:
    hidden = hidden @ previous[f'{layer}.weight'].T + previous[f'{layer}.bias']
    hidden = hidden/(1+np.exp(-hidden))
baseline = np.clip(hidden @ previous['4.weight'].T+previous['4.bias'],-2,2)
baseline[:,0] *= .5
targets = np.clip(data['targets'],.02,20)
ratio = targets/np.exp(baseline)
initial = float(np.mean(ratio[~training]+baseline[~training]))
best, best_penalty = initial, None
best_coefficients = np.zeros((design.shape[1],2))
diagnostics = {}
train_design, train_ratio = design[training],ratio[training]
for penalty in [.001,.01,.1]:
    def objective(flattened):
        coefficients = flattened.reshape(design.shape[1],2)
        raw = train_design @ coefficients
        correction = np.clip(raw,-.5,.5)
        scaled = train_ratio*np.exp(-correction)
        loss = np.mean(scaled+correction)+penalty*np.sum(coefficients**2)
        gradient = train_design.T @ ((1-scaled)*(np.abs(raw)<.5))/scaled.size+2*penalty*coefficients
        return float(loss),gradient.ravel()

    fitted = minimize(objective,np.zeros(design.shape[1]*2),method='L-BFGS-B',jac=True,
                      options={'maxiter':200,'ftol':1e-10})
    coefficients = fitted.x.reshape(design.shape[1],2)
    correction = np.clip(design[~training] @ coefficients,-.5,.5)
    loss = float(np.mean(ratio[~training]*np.exp(-correction)+baseline[~training]+correction))
    diagnostics[str(penalty)] = {'validation_loss':loss,'optimizer_success':bool(fitted.success),'iterations':int(fitted.nit)}
    if loss<best:
        best,best_penalty,best_coefficients = loss,penalty,coefficients
    print(penalty,diagnostics[str(penalty)],flush=True)
np.savez(AGENT_DIR / 'periodic_weights.npz',coefficients=best_coefficients)
metrics = {'initial_validation_loss':initial,'best_validation_loss':best,'penalty':best_penalty,
           'diagnostics':diagnostics,'target_cutoff':'2026-07-26','validation_start':'2026-07-01',
           'interpretation':'Smooth bounded positive scale only; no session gates or closures.'}
(AGENT_DIR / 'periodic_weights.json').write_text(json.dumps(metrics,indent=2))
print(metrics,flush=True)
