"""Plain JSON inference for the fixed residual tree and its matched ridge control."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np

@lru_cache(maxsize=4)
def parameters(name):
    return json.loads((Path(__file__).parent/name).read_text())

def correction(frame,asset,name='nonlinear_refit.json',kind='tree'):
    from research_model import residual_features
    p=parameters(name)
    features=residual_features(frame,asset,p['shape_table'])
    x=np.array([features[k] for k in p['features']])
    if kind=='ridge':
        r=p['ridge']
        return float(((x-r['mean'])/np.asarray(r['std']))@r['coef'])
    if kind!='tree':raise ValueError('Unknown residual model')
    x=x.astype(np.float32)
    result=p['offset']
    for tree in p['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        result+=p['learning_rate']*tree['value'][node]
    return float(result)
