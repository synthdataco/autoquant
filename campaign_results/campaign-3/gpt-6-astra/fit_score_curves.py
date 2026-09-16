"""Fit volatility corrections to precomputed, official CRPS response curves."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from predict import AGENT_DIR, load_context, score_paths, store_root
from research import CRYPTO, EQUITY, features, generate_variant

GRID = np.array([.4,.55,.7,.85,1.,1.2,1.45,1.8,2.3,3.0])


def build_asset(asset):
    dataset = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    assets = CRYPTO + EQUITY
    indices = np.flatnonzero(dataset['inputs'][:,-len(assets)+assets.index(asset)] == 1)
    rng = np.random.default_rng(420 + assets.index(asset))
    indices = indices[rng.random(len(indices)) < .25]
    first = sorted(store_root(asset).glob('date=*.parquet'))[0].stem.removeprefix('date=')
    prices = load_context(asset,pd.Timestamp(first,tz='UTC'),pd.Timestamp('2026-07-26',tz='UTC'))
    records = []
    for row_index in indices:
        timestamp = pd.Timestamp(int(dataset['timestamps'][row_index]),tz='UTC')
        context = prices.loc[(prices.index > timestamp-pd.Timedelta(days=7)) & (prices.index <= timestamp)]
        feature = features(context,timestamp,60)
        curves = np.full((2,len(GRID)),np.nan)
        for head,length,increment in [(0,3600,60),(1,86400,300)]:
            if head == 0 and asset not in CRYPTO:
                continue
            realized = prices.close.loc[timestamp:timestamp+pd.Timedelta(seconds=length):increment//60].to_numpy(dtype=float)
            if len(realized) != length//increment+1 or not np.isfinite(realized[-1]):
                continue
            base_paths = generate_variant(feature,asset,timestamp,increment,length,128,'auto:0:1:5')
            log_paths = np.log(base_paths/feature['last'])
            for grid_index,multiplier in enumerate(GRID):
                paths = feature['last'] * np.exp(log_paths*multiplier)
                curves[head,grid_index] = score_paths(paths,realized,increment,asset,length)[0]
        if np.isfinite(curves).any():
            records.append((int(row_index),curves))
    print(asset,'curve contexts',len(records),flush=True)
    return records


def fit(path):
    import torch
    from torch import nn

    torch.manual_seed(73)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    dataset=np.load(AGENT_DIR/'results'/'volatility_training.npz')
    curve_data=np.load(path)
    indices,curves=curve_data['indices'],curve_data['curves']
    inputs=dataset['inputs'][indices]
    timestamps=dataset['timestamps'][indices]
    trained=dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    values=np.clip((inputs-trained['center'])/trained['spread'],-8,8)
    valid=np.isfinite(curves).all(axis=2)&(curves[:,:,4]>0)
    normalized=np.nan_to_num(curves/np.maximum(curves[:,:,4:5],1e-6),nan=1.,posinf=1.,neginf=1.)
    training=timestamps<pd.Timestamp('2026-07-01',tz='UTC').value
    validation=~training
    net=nn.Sequential(nn.Linear(values.shape[1],48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    net.load_state_dict({name:torch.tensor(trained[name],device=device) for name in net.state_dict()})
    input_tensor=torch.tensor(values,dtype=torch.float32,device=device)
    curve_tensor=torch.tensor(normalized,dtype=torch.float32,device=device)
    valid_tensor=torch.tensor(valid,dtype=torch.float32,device=device)
    grid_tensor=torch.tensor(np.log(GRID),dtype=torch.float32,device=device)
    train_mask=torch.tensor(training,device=device)
    validation_mask=torch.tensor(validation,device=device)

    def objective(mask,penalty=0.):
        logits=net(input_tensor[mask]).clamp(float(np.log(GRID[0])),float(np.log(GRID[-1])))
        upper=torch.searchsorted(grid_tensor,logits.contiguous()).clamp(1,len(GRID)-1)
        lower=upper-1
        weight=(logits-grid_tensor[lower])/(grid_tensor[upper]-grid_tensor[lower])
        selected=curve_tensor[mask]
        low=torch.gather(selected,2,lower.unsqueeze(-1)).squeeze(-1)
        high=torch.gather(selected,2,upper.unsqueeze(-1)).squeeze(-1)
        losses=low*(1-weight)+high*weight+penalty*logits.square()
        valid_values=valid_tensor[mask]
        return (losses*valid_values).sum()/valid_values.sum()

    optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.05)
    best_loss,best_state,best_epoch=float('inf'),None,-1
    initial_loss=float(objective(validation_mask).detach().cpu())
    for epoch in range(300):
        optimizer.zero_grad()
        loss=objective(train_mask,.003)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss=float(objective(validation_mask).cpu())
        if validation_loss<best_loss:
            best_loss,best_epoch=validation_loss,epoch
            best_state={name:value.detach().cpu().numpy().copy() for name,value in net.state_dict().items()}
        if epoch%50==0:
            print('epoch',epoch,'train',float(loss.detach().cpu()),'validation',validation_loss,flush=True)
    np.savez(AGENT_DIR/'score_weights.npz',center=trained['center'],spread=trained['spread'],ridge=trained['ridge'],**best_state)
    metrics={'contexts':len(indices),'valid_heads':int(valid.sum()),'official_crps_evaluations':int(valid.sum()*len(GRID)),
             'paths_per_training_curve':128,'initial_interpolated_validation_ratio':initial_loss,
             'best_interpolated_validation_ratio':best_loss,'best_epoch':best_epoch,
             'target_cutoff':'2026-07-26','chronological_validation_start':'2026-07-01',
             'warning':'Interpolated training objective is not a final CRPS or field evaluation.'}
    (AGENT_DIR/'score_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    path=AGENT_DIR/'results'/'training_score_curves.npz'
    if not path.exists():
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results=list(executor.map(build_asset,CRYPTO+EQUITY))
        records=[record for batch in results for record in batch]
        np.savez(path,indices=np.array([record[0] for record in records]),curves=np.array([record[1] for record in records]))
    fit(path)


if __name__=='__main__':
    main()
