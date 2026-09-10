"""Fixed-policy fine-tuning using labels completed before an activation date."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR, ensure_workspace_cwd, load_context
from research import features
from train_vol import ASSETS, feature_vector

CUTOFF = pd.Timestamp('2026-08-07',tz='UTC')
FIRST_NEW = pd.Timestamp('2026-07-25',tz='UTC')


def make_asset(asset):
    prices = load_context(asset,FIRST_NEW-pd.Timedelta(days=7),CUTOFF)
    grid = pd.date_range(FIRST_NEW-pd.Timedelta(days=7),CUTOFF,freq='1min',inclusive='left')
    prices = prices.reindex(grid)
    close = prices.close.where(np.isfinite(prices.close)&(prices.close>0)).dropna()
    gaps = np.diff(close.index.as_unit('ns').asi8)/60e9
    absolute = pd.Series(np.abs(np.diff(np.log(close.to_numpy())))/np.sqrt(gaps),index=close.index[1:]).reindex(grid)
    targets = {horizon:absolute.shift(-1).iloc[::-1].rolling(horizon,min_periods=max(10,horizon//10)).mean().iloc[::-1]*np.sqrt(np.pi/2)
               for horizon in [60,1440]}
    rows,labels,stamps = [],[],[]
    for timestamp in pd.date_range(FIRST_NEW,CUTOFF-pd.Timedelta(days=1),freq='3h',inclusive='left'):
        context = prices.loc[(prices.index>timestamp-pd.Timedelta(days=7))&(prices.index<=timestamp)]
        if context.close.notna().sum()<50:
            continue
        target = np.array([targets[horizon].loc[timestamp] for horizon in [60,1440]])
        if not np.isfinite(target).all() or min(target)<=0:
            continue
        feature = features(context,timestamp,60)
        row = feature_vector(feature,asset)
        label = target/np.array([max(feature['av'][60],1e-8),max(feature['av'][360],1e-8)])
        if not np.isfinite(row).all() or not np.isfinite(label).all():
            continue
        rows.append(row)
        labels.append(label)
        stamps.append(timestamp.value)
    print(asset,len(rows),'new rows',flush=True)
    return np.asarray(rows),np.asarray(labels),np.asarray(stamps)


def main():
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    path = AGENT_DIR/'results'/'refresh_training.npz'
    if not path.exists():
        original = np.load(AGENT_DIR/'results'/'volatility_training.npz')
        keep = original['timestamps']<FIRST_NEW.value
        with ProcessPoolExecutor(max_workers=2) as executor:
            batches = list(executor.map(make_asset,ASSETS))
        np.savez(path,inputs=np.concatenate([original['inputs'][keep],*[batch[0] for batch in batches]]),
                 targets=np.concatenate([original['targets'][keep],*[batch[1] for batch in batches]]),
                 timestamps=np.concatenate([original['timestamps'][keep],*[batch[2] for batch in batches]]))
    data = np.load(path)
    assert data['timestamps'].max()+pd.Timedelta(days=1).value<CUTOFF.value
    previous = dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    standardized = np.clip((data['inputs']-previous['center'])/previous['spread'],-8,8)
    age = (CUTOFF.value-data['timestamps'])/86400e9
    weights = np.exp(-np.log(2)*age/60)
    weights /= weights.mean()
    torch.set_num_threads(3)
    torch.manual_seed(95)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    inputs = torch.tensor(standardized,dtype=torch.float32,device=device)
    targets = torch.tensor(np.clip(data['targets'],.02,20),dtype=torch.float32,device=device)
    weights = torch.tensor(weights,dtype=torch.float32,device=device)
    network = nn.Sequential(nn.Linear(30,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    network.load_state_dict({key:torch.tensor(previous[key],device=device) for key in network.state_dict()})
    optimizer = torch.optim.AdamW(network.parameters(),lr=.0005,weight_decay=.02)
    for epoch in range(40):
        for batch in torch.randperm(len(inputs),device=device).split(1024):
            output = network(inputs[batch]).clamp(-2,2)
            loss = ((targets[batch]*torch.exp(-output)+output+.03*output.square()).mean(dim=1)*weights[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    state = {key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
    np.savez(AGENT_DIR/'refresh_weights.npz',center=previous['center'],spread=previous['spread'],**state)
    metrics = {'contexts':len(inputs),'activation_date':str(CUTOFF),'all_target_endpoints_before':str(CUTOFF),
               'recency_half_life_days':60,'epochs':40,'seed':95,'initialization':'Original likelihood model',
               'last_batch_training_loss':float(loss.detach().cpu())}
    (AGENT_DIR/'refresh_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


if __name__=='__main__':
    main()
