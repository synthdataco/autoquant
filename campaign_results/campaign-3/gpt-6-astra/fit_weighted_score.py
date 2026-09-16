"""Fine-tune on raw official CRPS with validator asset weighting."""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR
from research import CRYPTO,EQUITY
from synth.validator.moving_average import ASSET_COEFFICIENTS

print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
torch.set_num_threads(4)
torch.manual_seed(73)
device='cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    torch.cuda.set_per_process_memory_fraction(.35)
data=np.load(AGENT_DIR/'results'/'volatility_training.npz')
curve_data=np.load(AGENT_DIR/'results'/'training_score_curves.npz')
indices,curves=curve_data['indices'],curve_data['curves']
inputs=data['inputs'][indices]
timestamps=data['timestamps'][indices]
previous=dict(np.load(AGENT_DIR/'volatility_weights.npz'))
assets=np.asarray(CRYPTO+EQUITY)[np.argmax(inputs[:,-13:],axis=1)]
valid=np.isfinite(curves).all(axis=2)&(curves[:,:,4]>0)
training=timestamps<pd.Timestamp('2026-07-01',tz='UTC').value
validation=~training
groups=[(0,np.isin(assets,CRYPTO)),(1,np.isin(assets,CRYPTO)),(1,np.isin(assets,EQUITY))]


def asset_weights(mask):
    weights=np.zeros(valid.shape)
    for head,group in groups:
        for asset in CRYPTO+EQUITY:
            selected=mask&group&(assets==asset)&valid[:,head]
            if selected.any():
                weights[selected,head]=ASSET_COEFFICIENTS[asset]/selected.sum()
        total=weights[group,head].sum()
        if total:
            weights[group,head]/=total
    return weights


train_weights=asset_weights(training)
validation_weights=asset_weights(validation)
normalized=np.nan_to_num(curves,nan=0.,posinf=0.,neginf=0.)
scales=[]
for head,group in groups:
    scale=float(np.sum(normalized[group,head,4]*train_weights[group,head]))
    normalized[group,head,:]/=max(scale,1e-6)
    scales.append(scale)
values=np.clip((inputs-previous['center'])/previous['spread'],-8,8)
input_tensor=torch.tensor(values,dtype=torch.float32,device=device)
curve_tensor=torch.tensor(normalized,dtype=torch.float32,device=device)
train_weight_tensor=torch.tensor(train_weights,dtype=torch.float32,device=device)
validation_weight_tensor=torch.tensor(validation_weights,dtype=torch.float32,device=device)
grid=torch.tensor(np.log([.4,.55,.7,.85,1.,1.2,1.45,1.8,2.3,3.0]),dtype=torch.float32,device=device)
network=nn.Sequential(nn.Linear(30,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
network.load_state_dict({key:torch.tensor(previous[key],device=device) for key in network.state_dict()})


def objective(weights,penalty=0.):
    logits=network(input_tensor).clamp(float(grid[0].cpu()),float(grid[-1].cpu()))
    upper=torch.searchsorted(grid,logits.contiguous()).clamp(1,len(grid)-1)
    lower=upper-1
    fraction=(logits-grid[lower])/(grid[upper]-grid[lower])
    low=torch.gather(curve_tensor,2,lower.unsqueeze(-1)).squeeze(-1)
    high=torch.gather(curve_tensor,2,upper.unsqueeze(-1)).squeeze(-1)
    loss=low*(1-fraction)+high*fraction+penalty*logits.square()
    return (loss*weights).sum()/weights.sum()


optimizer=torch.optim.AdamW(network.parameters(),lr=.0003,weight_decay=.05)
initial=float(objective(validation_weight_tensor).detach().cpu())
best,best_epoch=initial,-1
best_state={key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
for epoch in range(300):
    optimizer.zero_grad()
    loss=objective(train_weight_tensor,.003)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        validation_loss=float(objective(validation_weight_tensor).cpu())
    if validation_loss<best:
        best,best_epoch=validation_loss,epoch
        best_state={key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
    if epoch%50==0:
        print('epoch',epoch,'validation',validation_loss,'best',best,flush=True)
np.savez(AGENT_DIR/'weighted_score_weights.npz',center=previous['center'],spread=previous['spread'],**best_state)
metrics={'initial_weighted_validation_loss':initial,'best_weighted_validation_loss':best,'best_epoch':best_epoch,
         'constant_competition_scales':scales,'target_cutoff':'2026-07-26','validation_start':'2026-07-01',
         'objective':'Raw official CRPS, asset-coefficient and asset-frequency weighting; constant group scaling only',
         'paths_per_training_curve':128}
(AGENT_DIR/'weighted_score_weights.json').write_text(json.dumps(metrics,indent=2))
print(metrics,flush=True)
