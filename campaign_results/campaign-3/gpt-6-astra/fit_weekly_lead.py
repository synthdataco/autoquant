"""Chronological fit of a smooth weekly residual around frozen lead models."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from predict import AGENT_DIR, ensure_workspace_cwd
from temporal import temporal_embedding
from weekly_lead import weekly_design


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    sequences = np.load(AGENT_DIR/'results'/'temporal_exact_training.npz')
    leads = np.load(AGENT_DIR/'results'/'lead_training.npz')
    selected = sequences['indices']
    inputs = data['inputs'][selected]
    keys = pd.MultiIndex.from_arrays([data['timestamps'][selected],inputs[:,-13:].argmax(axis=1)])
    lead_keys = pd.MultiIndex.from_arrays([leads['timestamps'],leads['inputs'][:,-13:].argmax(axis=1)])
    positions = lead_keys.get_indexer(keys)
    valid = positions>=0
    inputs = inputs[valid]
    targets = np.clip(leads['targets'][positions[valid]],.02,20)
    timestamps = pd.to_datetime(data['timestamps'][selected][valid],utc=True)
    assets = inputs[:,-13:].argmax(axis=1)
    weights = dict(np.load(AGENT_DIR/'lead_weights.npz'))
    static = np.clip((inputs-weights['center'])/weights['spread'],-8,8)
    hidden = static.copy()
    for layer in [0,2]:
        hidden = hidden @ weights[f'{layer}.weight'].T+weights[f'{layer}.bias']
        hidden = hidden/(1+np.exp(-hidden))
    baseline = np.clip(hidden @ weights['4.weight'].T+weights['4.bias'],-2,2).astype(float)
    embedded = np.stack([temporal_embedding(sequence,values,'temporal_exact')
                         for sequence,values in zip(sequences['sequences'][valid],inputs)])
    linear = dict(np.load(AGENT_DIR/'lead_temporal_weights.npz'))
    linear_design = np.column_stack([embedded,static])
    linear_design = np.column_stack([np.ones(len(inputs)),np.clip((linear_design-linear['center'])/linear['spread'],-8,8)])
    duration = np.diff(weights['edges'])/weights['edges'][-1]
    raw = baseline+np.clip(linear_design @ linear['coefficients'],-.5,.5)
    baseline = raw-np.log((np.exp(raw)*duration).sum(axis=1)/(np.exp(baseline)*duration).sum(axis=1))[:,None]
    fine = dict(np.load(AGENT_DIR/'lead_finetune_weights.npz'))
    equity = assets>=5
    embedded_fine = np.stack([temporal_embedding(sequence,values,'lead_finetune')
                              for sequence,values in zip(sequences['sequences'][valid][equity],inputs[equity])])
    raw = baseline[equity]+np.clip(embedded_fine @ fine['head.2.weight'].T+fine['head.2.bias'],-.5,.5)
    baseline[equity] = raw-np.log((np.exp(raw)*duration).sum(axis=1)/(np.exp(baseline[equity])*duration).sum(axis=1))[:,None]
    design = weekly_design(timestamps,assets,weights['edges'])
    training = timestamps<pd.Timestamp('2026-07-01',tz='UTC')

    def loss_gradient(coefficients, mask, penalty):
        selected_design = design[mask]
        raw_correction = selected_design @ coefficients
        correction = np.clip(raw_correction,-.25,.25)
        raw = baseline[mask]+correction
        density = np.exp(raw)*duration
        totals = density.sum(axis=1,keepdims=True)
        normalization = np.log(totals/(np.exp(baseline[mask])*duration).sum(axis=1,keepdims=True))
        prediction = raw-normalization
        scaled = targets[mask]*np.exp(-prediction)
        loss = ((scaled+prediction)*duration).sum(axis=1).mean()+penalty*np.sum(coefficients**2)
        gradient = (1-scaled)*duration/len(raw)
        gradient -= gradient.sum(axis=1,keepdims=True)*density/totals
        gradient *= np.abs(raw_correction)<.25
        derivative = np.einsum('nbf,nb->f',selected_design,gradient)+2*penalty*coefficients
        return float(loss),derivative

    best_coefficients = np.zeros(design.shape[-1])
    initial = best = loss_gradient(best_coefficients,~training,0)[0]
    best_penalty = None
    diagnostics = {}
    for penalty in [.01,.1,1.]:
        fit = minimize(lambda coefficients:loss_gradient(coefficients,training,penalty),
                       np.zeros(design.shape[-1]),jac=True,method='L-BFGS-B',
                       options={'maxiter':150,'ftol':1e-10})
        loss = loss_gradient(fit.x,~training,0)[0]
        diagnostics[str(penalty)] = {'loss':loss,'success':bool(fit.success),'iterations':int(fit.nit)}
        print(penalty,diagnostics[str(penalty)],flush=True)
        if loss<best:
            best,best_penalty,best_coefficients = loss,penalty,fit.x
    np.savez(AGENT_DIR/'weekly_lead_weights.npz',coefficients=best_coefficients,
             edges=weights['edges'],duration=duration)
    metrics = {'initial_validation_loss':initial,'best_validation_loss':best,'penalty':best_penalty,
               'diagnostics':diagnostics,'training_rows':int(training.sum()),'validation_rows':int((~training).sum()),
               'target_cutoff':'2026-07-26','validation_start':'2026-07-01','passed_gate':bool(initial-best>=.0001)}
    (AGENT_DIR/'weekly_lead_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


if __name__=='__main__':
    main()
