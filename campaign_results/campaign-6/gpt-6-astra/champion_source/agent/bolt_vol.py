"""Offline Chronos-Bolt median log-variance experiment, CPU only."""
from functools import lru_cache
from pathlib import Path
import os,random
import numpy as np,pandas as pd
os.environ['TORCH_DISABLE_NATIVE_JIT']='1'

@lru_cache(maxsize=1)
def pipeline():
    import torch
    from chronos import ChronosBoltPipeline
    torch.set_num_threads(1)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    return ChronosBoltPipeline.from_pretrained(str(Path(__file__).parent/'models'/'chronos-bolt-tiny'),device_map='cpu',dtype=torch.float32,local_files_only=True)

@lru_cache(maxsize=2048)
def predict_hourly(asset,origin,seed,values):
    import torch
    random.seed(seed);torch.manual_seed(seed)
    x=np.asarray(values,dtype=float);center=float(np.median(x))
    series=np.log(np.maximum(x,center*.001)/center)
    model=pipeline()
    random.seed(seed);torch.manual_seed(seed)
    with torch.inference_mode():
        quantiles,_=model.predict_quantiles(torch.tensor(series,dtype=torch.float32),prediction_length=25,quantile_levels=[.5])
    point=quantiles.detach().cpu().numpy()[0,:,0]
    return np.exp(np.clip(point,-3,3))*center

def forecast_variance(frame,asset,future,seed,shape_name='seasonality.json'):
    from research_model import shape_at
    ret=np.log(frame.close.astype(float)).diff()
    rv=ret.pow(2)/shape_at(frame.index,asset,True,shape_name)
    hourly=rv.resample('h').mean();count=frame.close.resample('h').size()
    hourly=hourly.where(count==60)
    hourly=hourly[hourly.index+pd.Timedelta(hours=1)<=frame.index[-1]+pd.Timedelta(minutes=1)].iloc[-144:]
    bad=np.flatnonzero(~np.isfinite(hourly.to_numpy()))
    if len(bad):hourly=hourly.iloc[bad[-1]+1:]
    if len(hourly)<48 or float(hourly.median())<=0:return None
    origin=hourly.index[-1]+pd.Timedelta(hours=1)
    forecast=predict_hourly(asset,str(origin),seed,tuple(hourly.to_numpy(dtype=float)))
    at=np.asarray((future-origin).total_seconds())//3600
    return forecast[np.clip(at.astype(int),0,len(forecast)-1)]
