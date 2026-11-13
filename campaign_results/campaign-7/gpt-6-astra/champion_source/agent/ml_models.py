"""Learned scale adjustment; seasonal Gaussian paths otherwise unchanged."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
from ml_features import NAMES, one_features
from portable_trees import portable_booster
from seasonal_models import forecast_variance

CONTEXT_DAYS=7
VARIANT=os.environ.get('SYNTH_RESEARCH_VARIANT','ml_vol')
ARTIFACTS=Path(__file__).resolve().parent/'artifacts/ml'

@lru_cache(maxsize=1)
def manifest():
    return json.loads((ARTIFACTS/'manifest.json').read_text())

@lru_cache(maxsize=32)
def booster(filename):
    # Retained for research/export utilities; the serving path is NumPy-only.
    import xgboost as xgb
    model=xgb.Booster(params={'device':'cpu','nthread':1})
    model.load_model(ARTIFACTS/filename)
    model.set_param({'device':'cpu','nthread':1})
    return model

def scale_adjustment(asset,start_time,time_length,context,kind='vol',blend=1.0):
    date=str(context.index[-1].date())
    fits=[x for x in manifest()['fits'] if x['cutoff']<=date and x['time_length']==time_length and x['kind']==kind]
    if not fits:
        return 1.0
    fit=max(fits,key=lambda x:x['cutoff'])
    x,_=one_features(asset,start_time,time_length,context)
    x=x[[NAMES.index(k) for k in fit['features']]].reshape(1,-1)
    x[~np.isfinite(x)]=np.nan
    prediction=float(portable_booster(fit['file']).inplace_predict(x)[0])
    return float(np.exp(blend*np.clip(prediction,np.log(.25),np.log(3.0))))

def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    rng=np.random.default_rng(seed)
    last,stepvar=forecast_variance(asset,start_time,time_increment,time_length,context_prices)
    kind='full' if 'full' in VARIANT else 'vol'
    blend=.5 if 'blend' in VARIANT else 1.
    adjustment=scale_adjustment(asset,start_time,time_length,context_prices,kind,blend)
    steps=time_length//time_increment+1
    z=rng.normal(size=(num_simulations,steps))
    z[:,0]=0
    z[:,1:]*=np.sqrt(stepvar)*adjustment
    paths=last*np.exp(np.cumsum(z,axis=1))
    return (start_time,time_increment,*paths.tolist())
