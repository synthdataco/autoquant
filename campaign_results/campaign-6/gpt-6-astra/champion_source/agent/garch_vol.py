"""Variance-targeted GARCH experiment; no prices are imputed."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from numba import njit

def finite_ewma(values,half=4320,window=10080):
    values=np.asarray(values,dtype=float);ok=np.isfinite(values);rate=2**(-1/half)
    num=lfilter([1.],[1.,-rate],np.where(ok,values,0.));den=lfilter([1.],[1.,-rate],ok.astype(float))
    if len(values)>window:
        num[window:]-=rate**window*num[:-window].copy()
        den[window:]-=rate**window*den[:-window].copy()
    return np.divide(num,den,out=np.full(len(values),1e-8),where=den>1e-12)

@njit(cache=True)
def filter_state(normalized,alpha,persistence,loss_mask):
    state=1.;loss=0.;n=0
    for i in range(len(normalized)):
        value=normalized[i]
        state=max(state,1e-10)
        if np.isfinite(value):
            if loss_mask[i]:loss+=np.log(state)+value/state;n+=1
            state=(1-persistence)+alpha*value+(persistence-alpha)*state
        else:
            state=(1-persistence)+persistence*state
    return state,loss/max(n,1)

def prepare(frame,asset,shape_name='seasonality.json',clock=False):
    from research_model import shape_at,clock_at
    close=frame.close.astype(float)
    close=close.reindex(pd.date_range(close.index[0],close.index[-1],freq='min'))
    shape=shape_at(close.index,asset,True,shape_name)
    if clock:shape*=clock_at(close.index,asset,1)
    values=np.log(close).diff().to_numpy()**2/shape
    level=finite_ewma(values)
    previous=np.concatenate([np.array([1e-8]),level[:-1]])
    return close.index,values,level,values/np.maximum(previous,1e-14)

@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).parent/'garch.json').read_text())['assets']

def forecast(frame,asset,future,shape_name='seasonality.json',clock=False):
    index,values,level,z=prepare(frame,asset,shape_name,clock)
    fitted=parameters()[asset];state,_=filter_state(z,fitted['alpha'],fitted['persistence'],np.zeros(len(z),dtype=np.bool_))
    elapsed=np.asarray((future-index[-1]).total_seconds())/60
    return level[-1]*(1+(state-1)*fitted['persistence']**np.maximum(elapsed-1,0))
