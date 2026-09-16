"""Refit the fixed likelihood architecture using all pre-validation labels."""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from predict import AGENT_DIR

print((AGENT_DIR/'BUDGET.md').read_text(),flush=True)
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    torch.cuda.set_per_process_memory_fraction(.35)
data=np.load(AGENT_DIR/'results'/'volatility_training.npz')
previous=dict(np.load(AGENT_DIR/'volatility_weights.npz'))
values=np.clip((data['inputs']-previous['center'])/previous['spread'],-8,8)
age=(pd.Timestamp('2026-07-26',tz='UTC').value-data['timestamps'])/86400e9
inputs=torch.tensor(values,dtype=torch.float32,device=device)
targets=torch.tensor(np.clip(data['targets'],.02,20),dtype=torch.float32,device=device)
for name,half_life in [('uniform',None),('recent60',60),('recent120',120)]:
    torch.manual_seed(95)
    weights=np.ones(len(age)) if half_life is None else np.exp(-np.log(2)*age/half_life)
    weights=weights/weights.mean()
    weight_tensor=torch.tensor(weights,dtype=torch.float32,device=device)
    network=nn.Sequential(nn.Linear(values.shape[1],48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    network.load_state_dict({key:torch.tensor(previous[key],device=device) for key in network.state_dict()})
    optimizer=torch.optim.AdamW(network.parameters(),lr=.0005,weight_decay=.02)
    for epoch in range(40):
        order=torch.randperm(len(inputs),device=device)
        for batch in order.split(1024):
            output=network(inputs[batch]).clamp(-2,2)
            loss=((targets[batch]*torch.exp(-output)+output+.03*output.square()).mean(dim=1)*weight_tensor[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    state={key:value.detach().cpu().numpy().copy() for key,value in network.state_dict().items()}
    np.savez(AGENT_DIR/f'refit_{name}_weights.npz',center=previous['center'],spread=previous['spread'],ridge=previous['ridge'],**state)
    metadata={'contexts':len(inputs),'target_cutoff':'2026-07-26','recency_half_life_days':half_life,'epochs':40,'initialization':'Original likelihood model','last_batch_training_loss':float(loss.detach().cpu())}
    (AGENT_DIR/f'refit_{name}_weights.json').write_text(json.dumps(metadata,indent=2))
    print(name,metadata,flush=True)
