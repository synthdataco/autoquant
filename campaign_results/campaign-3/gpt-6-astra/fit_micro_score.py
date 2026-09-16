"""Matched multi-minute ablation using only official finite-fan CRPS."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fit_newton_score import weights_for
from micro_score import micro_design
from predict import AGENT_DIR, ensure_workspace_cwd, load_context, score_paths
from research import CRYPTO, EQUITY, features, generate_variant
from synth_lib.backtester.preparation import _slice_real_prices


def build_asset(task):
    asset,verify,fingerprint = task
    ensure_workspace_cwd()
    checkpoint = AGENT_DIR/'results'/'micro_score.parts'/f'{asset}_{fingerprint}.npz'
    if checkpoint.exists() and not verify:
        return dict(np.load(checkpoint))
    original = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    indices = np.load(AGENT_DIR/'results'/'training_score_curves.npz')['indices']
    indices = indices[original['inputs'][indices,-13:].argmax(axis=1)==(CRYPTO+EQUITY).index(asset)]
    if verify:
        indices = indices[original['timestamps'][indices]>=pd.Timestamp('2026-07-01',tz='UTC').value]
    timestamps = pd.to_datetime(original['timestamps'][indices],utc=True)
    cutoff = pd.Timestamp('2026-07-26',tz='UTC')
    prices = load_context(asset,timestamps.min()-pd.Timedelta(days=7),cutoff)
    variants = ['stackmicro:constant:1','stackmicro:control','stackmicro:extra'] if verify else [f'stackmicro:constant:{value}' for value in [.92,1.,1.08]]
    records = []
    for index,timestamp in zip(indices,timestamps):
        if timestamp+pd.Timedelta(days=1)>cutoff:
            continue
        context = prices.loc[(prices.index>timestamp-pd.Timedelta(days=7)) & (prices.index<=timestamp)]
        feature = features(context,timestamp,300)
        realized = _slice_real_prices(prices[['close']],timestamp,86400,300)
        if not np.isfinite(realized[0]) or not np.isfinite(realized[-1]):
            continue
        scores = [score_paths(generate_variant(feature,asset,timestamp,300,86400,1000,variant),
                              realized,300,asset,86400)[0] for variant in variants]
        if np.isfinite(scores).all() and min(scores)>0:
            records.append((int(index),micro_design(feature,asset,timestamp),scores))
    result = {'indices':np.array([row[0] for row in records]),'design':np.stack([row[1] for row in records]),
              'scores':np.array([row[2] for row in records])}
    if not verify:
        checkpoint.parent.mkdir(exist_ok=True)
        temporary = checkpoint.with_suffix('.tmp.npz')
        np.savez(temporary,**result)
        temporary.replace(checkpoint)
    print(asset,len(records),'verified' if verify else 'curves',flush=True)
    return result


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    fingerprint = hashlib.sha256()
    for source in sorted([*AGENT_DIR.glob('*.py'),*[AGENT_DIR/f'{kind}_weights.npz' for kind in
                          ['volatility','temporal_exact','lead','lead_temporal','tree_student','periodic','lead_finetune']]]):
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    digest = fingerprint.hexdigest()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        batches = list(executor.map(build_asset,[(asset,False,digest) for asset in CRYPTO+EQUITY]))
    stored = {key:np.concatenate([batch[key] for batch in batches]) for key in ['indices','design','scores']}
    np.savez(AGENT_DIR/'results'/'micro_score_training.npz',**stored)
    original = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    assets = np.asarray(CRYPTO+EQUITY)[original['inputs'][stored['indices'],-13:].argmax(axis=1)]
    training = original['timestamps'][stored['indices']]<pd.Timestamp('2026-07-01',tz='UTC').value
    scores = stored['scores']
    gradient = (scores[:,2]-scores[:,0])/.16
    curvature = (scores[:,2]-2*scores[:,1]+scores[:,0])/.08**2
    floor = max(float(np.median(curvature[training]))*.001,1e-6)
    curvature = np.maximum(curvature,floor)
    train_weights,normalizers = weights_for(assets,training,scores[:,1])
    diagnostic_weights,_ = weights_for(assets,~training,scores[:,1],normalizers)
    diagnostics = {}
    choices = {}
    for kind,raw_design in [('control',stored['design'][:,:-9]),('extra',stored['design'])]:
        center = raw_design[training].mean(axis=0)
        spread = np.maximum(raw_design[training].std(axis=0),.1)
        design = np.column_stack([np.ones(len(assets)),np.clip((raw_design-center)/spread,-8,8)])
        matrix = design.T @ ((train_weights*curvature)[:,None]*design)
        direction = -(design.T @ (train_weights*gradient))
        best,best_penalty = 0.,None
        coefficients = np.zeros(design.shape[1])
        for penalty in [.001,.01,.1,1.]:
            candidate = np.linalg.solve(matrix+penalty*np.eye(design.shape[1]),direction)
            change = np.clip(design @ candidate,-.08,.08)
            surrogate = float(diagnostic_weights @ (gradient*change+.5*curvature*change**2))
            diagnostics[f'{kind}/{penalty}'] = surrogate
            print(kind,penalty,surrogate,flush=True)
            if surrogate<best:
                best,best_penalty,coefficients = surrogate,penalty,candidate
        np.savez(AGENT_DIR/f'micro_{kind}_weights.npz',center=center,spread=spread,coefficients=coefficients)
        choices[kind] = {'surrogate_change':best,'penalty':best_penalty}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        batches = list(executor.map(build_asset,[(asset,True,digest) for asset in CRYPTO+EQUITY]))
    verified = {key:np.concatenate([batch[key] for batch in batches]) for key in ['indices','scores']}
    np.savez(AGENT_DIR/'results'/'micro_score_verification.npz',**verified)
    verified_assets = np.asarray(CRYPTO+EQUITY)[original['inputs'][verified['indices'],-13:].argmax(axis=1)]
    weights,_ = weights_for(verified_assets,np.ones(len(verified_assets),dtype=bool),verified['scores'][:,0],normalizers)
    ratios = {}
    for name,group in [('crypto',CRYPTO),('com_equ',EQUITY)]:
        selected = np.isin(verified_assets,group)
        totals = weights[selected] @ verified['scores'][selected]
        ratios[name] = {'control_identity':float(totals[1]/totals[0]),'extra_identity':float(totals[2]/totals[0]),
                        'extra_control':float(totals[2]/totals[1]),
                        'passed_gate':bool(totals[2]/totals[0]<.9999 and totals[2]/totals[1]<.9999)}
    metrics = {'diagnostics':diagnostics,'choices':choices,'verification_ratios':ratios,'curvature_floor':floor,
               'training_rows':int(training.sum()),'validation_rows':int((~training).sum()),'paths':1000,
               'target_cutoff':'2026-07-26','validation_start':'2026-07-01','fingerprint':digest}
    (AGENT_DIR/'micro_score_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


if __name__=='__main__':
    main()
