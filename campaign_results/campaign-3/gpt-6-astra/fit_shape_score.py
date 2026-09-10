"""Official-score response curves matched to the incumbent path shape."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from predict import AGENT_DIR, ensure_workspace_cwd, load_context, score_paths
from research import CRYPTO, EQUITY, features, generate_variant
from synth_lib.backtester.preparation import _slice_real_prices
from synth.validator.moving_average import ASSET_COEFFICIENTS

GRID = np.array([.7,.85,1.,1.2,1.45])


def build_asset(asset):
    ensure_workspace_cwd()
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    prior = np.load(AGENT_DIR/'results'/'training_score_curves.npz')
    indices = prior['indices']
    asset_index = (CRYPTO+EQUITY).index(asset)
    indices = indices[np.argmax(data['inputs'][indices,-13:],axis=1)==asset_index]
    stamps = pd.to_datetime(data['timestamps'][indices],utc=True)
    prices = load_context(asset,stamps.min()-pd.Timedelta(days=7),pd.Timestamp('2026-07-26',tz='UTC'))
    records = []
    for index,timestamp in zip(indices,stamps):
        context = prices.loc[timestamp-pd.Timedelta(days=7):timestamp]
        context = context.loc[context.index>timestamp-pd.Timedelta(days=7)]
        feature = features(context,timestamp,60)
        curves = np.full((2,len(GRID)),np.nan)
        for head,length,increment in [(0,3600,60),(1,86400,300)]:
            if head==0 and asset not in CRYPTO:
                continue
            realized = _slice_real_prices(prices[['close']],timestamp,length,increment)
            if len(realized)!=length//increment+1 or not np.isfinite(realized[-1]):
                continue
            paths = generate_variant(feature,asset,timestamp,increment,length,1000,'lead:0.5')
            logs = np.log(paths/feature['last'])
            for position,multiplier in enumerate(GRID):
                curves[head,position] = score_paths(feature['last']*np.exp(logs*multiplier),realized,increment,asset,length)[0]
        if np.isfinite(curves).any():
            records.append((int(index),curves))
    print(asset,len(records),'shape-matched contexts',flush=True)
    return records


def fit(path, name='shape_score', epochs=300):
    import torch
    from torch import nn

    torch.set_num_threads(3)
    torch.manual_seed(73)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.35)
    data = np.load(AGENT_DIR/'results'/'volatility_training.npz')
    stored = np.load(path)
    indices,curves = stored['indices'],stored['curves']
    previous = dict(np.load(AGENT_DIR/'volatility_weights.npz'))
    inputs = data['inputs'][indices]
    assets = np.asarray(CRYPTO+EQUITY)[np.argmax(inputs[:,-13:],axis=1)]
    training = data['timestamps'][indices]<pd.Timestamp('2026-07-01',tz='UTC').value
    center_position = int(np.flatnonzero(GRID==1)[0])
    valid = np.isfinite(curves).all(axis=2)&(curves[:,:,center_position]>0)
    groups = [(0,np.isin(assets,CRYPTO)),(1,np.isin(assets,CRYPTO)),(1,np.isin(assets,EQUITY))]

    def objective_weights(mask):
        result = np.zeros(valid.shape)
        for head,group in groups:
            for asset in CRYPTO+EQUITY:
                selected = mask&group&(assets==asset)&valid[:,head]
                if selected.any():
                    result[selected,head] = ASSET_COEFFICIENTS[asset]/selected.sum()
            total = result[group,head].sum()
            if total:
                result[group,head] /= total
        return result

    train_weights,validation_weights = objective_weights(training),objective_weights(~training)
    normalized = np.nan_to_num(curves,nan=0.,posinf=0.,neginf=0.)
    scales = []
    for head,group in groups:
        scale = float(np.sum(normalized[group,head,center_position]*train_weights[group,head]))
        normalized[group,head,:] /= max(scale,1e-6)
        scales.append(scale)
    values = torch.tensor(np.clip((inputs-previous['center'])/previous['spread'],-8,8),dtype=torch.float32,device=device)
    responses = torch.tensor(normalized,dtype=torch.float32,device=device)
    grid = torch.tensor(np.log(GRID),dtype=torch.float32,device=device)
    train_weights = torch.tensor(train_weights,dtype=torch.float32,device=device)
    validation_weights = torch.tensor(validation_weights,dtype=torch.float32,device=device)
    network = nn.Sequential(nn.Linear(30,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    with torch.no_grad():
        for layer in [0,2]:
            network[layer].weight.copy_(torch.tensor(previous[f'{layer}.weight'],device=device))
            network[layer].bias.copy_(torch.tensor(previous[f'{layer}.bias'],device=device))
        network[4].weight.zero_()
        network[4].bias.zero_()

    def objective(weights,penalty=0.):
        logs = network(values).clamp(float(np.log(GRID[0])),float(np.log(GRID[-1])))
        upper = torch.searchsorted(grid,logs.contiguous()).clamp(1,len(grid)-1)
        lower = upper-1
        fraction = (logs-grid[lower])/(grid[upper]-grid[lower])
        low = torch.gather(responses,2,lower.unsqueeze(-1)).squeeze(-1)
        high = torch.gather(responses,2,upper.unsqueeze(-1)).squeeze(-1)
        losses = low*(1-fraction)+high*fraction+penalty*logs.square()
        return (losses*weights).sum()/weights.sum()

    optimizer = torch.optim.AdamW(network.parameters(),lr=.0003,weight_decay=.05)
    initial = float(objective(validation_weights).detach().cpu())
    best,best_epoch = initial,-1
    best_state = {key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
    for epoch in range(epochs):
        optimizer.zero_grad()
        objective(train_weights,.01).backward()
        optimizer.step()
        with torch.no_grad():
            loss = float(objective(validation_weights).cpu())
        if loss<best:
            best,best_epoch = loss,epoch
            best_state = {key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
        if epoch%50==0:
            print(epoch,loss,best,flush=True)
    np.savez(AGENT_DIR/f'{name}_weights.npz',center=previous['center'],spread=previous['spread'],limits=np.log(GRID[[0,-1]]),**best_state)
    metrics = {'initial_validation_objective':initial,'best_validation_objective':best,'best_epoch':best_epoch,
               'paths':1000,'contexts':len(indices),'official_crps_calls':int(valid.sum()*len(GRID)),
               'target_cutoff':'2026-07-26','validation_start':'2026-07-01','constant_competition_scales':scales,
               'base_family':'lead:0.5; training response curves, not causal historical forecasts','scale_grid':GRID.tolist(),'epochs':epochs}
    (AGENT_DIR/f'{name}_weights.json').write_text(json.dumps(metrics,indent=2))
    print(metrics,flush=True)


def main():
    global GRID
    ensure_workspace_cwd()
    print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--fine',action='store_true')
    parser.add_argument('--epochs',type=int,default=300)
    args = parser.parse_args()
    if args.fine:
        GRID = np.array([.85,.9,.95,.975,1.,1.025,1.05,1.1,1.2])
    name = 'shape_score_fine' if args.fine else 'shape_score'
    path = AGENT_DIR/'results'/f'{name}_curves.npz'
    if not path.exists():
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(build_asset,CRYPTO+EQUITY))
        records = [record for batch in batches for record in batch]
        np.savez(path,indices=np.array([record[0] for record in records]),curves=np.array([record[1] for record in records]))
    fit(path,name if args.epochs==300 else f'{name}_{args.epochs}',args.epochs)


if __name__=='__main__':
    main()
