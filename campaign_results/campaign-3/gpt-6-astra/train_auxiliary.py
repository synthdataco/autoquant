"""Test whether optional OHLCV information adds to close-based forecasting."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from auxiliary import HALF_LIVES,auxiliary_features
from predict import AGENT_DIR,load_context,store_root
from research import CRYPTO,EQUITY


def make_asset(asset):
    data=np.load(AGENT_DIR/'results'/'volatility_training.npz')
    assets=CRYPTO+EQUITY
    indices=np.flatnonzero(data['inputs'][:,-len(assets)+assets.index(asset)]==1)
    first=sorted(store_root(asset).glob('date=*.parquet'))[0].stem.removeprefix('date=')
    prices=load_context(asset,pd.Timestamp(first,tz='UTC'),pd.Timestamp('2026-07-26',tz='UTC'))
    rows,available=[],[]
    for row in indices:
        timestamp=pd.Timestamp(int(data['timestamps'][row]),tz='UTC')
        context=prices.loc[(prices.index>timestamp-pd.Timedelta(days=7))&(prices.index<=timestamp)]
        reference=float(np.exp(data['inputs'][row,16]))
        absolute={half_life:float(np.exp(data['inputs'][row,column])*reference) for column,half_life in enumerate(HALF_LIVES)}
        features,usable=auxiliary_features(context,timestamp,reference,absolute)
        rows.append(features)
        available.append(usable)
    print(asset,len(rows),sum(available),flush=True)
    return indices,np.asarray(rows),np.asarray(available)


def train(path):
    import torch
    from torch import nn

    torch.set_num_threads(4)
    torch.manual_seed(91)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    data=np.load(AGENT_DIR/'results'/'volatility_training.npz')
    auxiliary=np.load(path)
    previous=dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    training=data['timestamps']<pd.Timestamp('2026-07-01',tz='UTC').value
    validation=~training
    inputs=np.column_stack([data['inputs'],auxiliary['features']])
    center=np.r_[previous['center'],auxiliary['features'][training].mean(axis=0)]
    spread=np.r_[previous['spread'],np.maximum(auxiliary['features'][training].std(axis=0),.1)]
    values=np.clip((inputs-center)/spread,-8,8)
    inputs_tensor=torch.tensor(values,dtype=torch.float32,device=device)
    targets=torch.tensor(np.clip(data['targets'],.02,20),dtype=torch.float32,device=device)
    network=nn.Sequential(nn.Linear(values.shape[1],48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    state=network.state_dict()
    for key in state:
        if key=='0.weight':
            state[key].zero_()
            state[key][:,:30]=torch.tensor(previous[key],device=device)
        else:
            state[key]=torch.tensor(previous[key],device=device)
    network.load_state_dict(state)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.02)
    train_indices=torch.tensor(np.flatnonzero(training),device=device)
    valid_indices=torch.tensor(np.flatnonzero(validation),device=device)
    best,best_state,best_epoch=float('inf'),None,-1
    for epoch in range(150):
        order=train_indices[torch.randperm(len(train_indices),device=device)]
        for batch in order.split(1024):
            logits=network(inputs_tensor[batch]).clamp(-2,2)
            loss=(targets[batch]*torch.exp(-logits)+logits+.03*logits.square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits=network(inputs_tensor[valid_indices]).clamp(-2,2)
            value=float((targets[valid_indices]*torch.exp(-logits)+logits).mean().cpu())
        if value<best:
            best,best_epoch=value,epoch
            best_state={key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
        if epoch%25==0:
            print('epoch',epoch,'validation',value,flush=True)
    np.savez(AGENT_DIR/'aux_weights.npz',center=center,spread=spread,**best_state)
    metrics={'best_validation_likelihood':best,'best_epoch':best_epoch,'inputs':inputs.shape[1],
             'training_contexts':int(training.sum()),'validation_contexts':int(validation.sum()),
             'target_cutoff':'2026-07-26','validation_start':'2026-07-01'}
    (AGENT_DIR/'aux_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


if __name__=='__main__':
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    path=AGENT_DIR/'results'/'auxiliary_training.npz'
    if not path.exists():
        data=np.load(AGENT_DIR/'results'/'volatility_training.npz')
        all_features=np.zeros((len(data['inputs']),19),dtype=np.float32)
        usable=np.zeros(len(all_features),dtype=bool)
        with ProcessPoolExecutor(max_workers=6) as executor:
            for indices,features,available in executor.map(make_asset,CRYPTO+EQUITY):
                all_features[indices]=features
                usable[indices]=available
        np.savez(path,features=all_features,usable=usable)
    train(path)
