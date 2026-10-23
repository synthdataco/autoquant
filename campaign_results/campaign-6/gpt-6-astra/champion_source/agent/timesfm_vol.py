"""Offline TimesFM-v1 point-log-variance study using its unchanged public decoder."""
from functools import lru_cache
from pathlib import Path
import os,random
import numpy as np,pandas as pd
os.environ['TORCH_DISABLE_NATIVE_JIT']='1'

@lru_cache(maxsize=1)
def pipeline():
    import torch
    from vendor.timesfm_v1.pytorch_patched_decoder import TimesFMConfig,PatchedTimeSeriesDecoder
    torch.set_num_threads(1)
    model=PatchedTimeSeriesDecoder(TimesFMConfig())
    checkpoint=Path(__file__).parent/'models'/'timesfm-1.0-200m'/'torch_model.ckpt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    return model.float().eval()

@lru_cache(maxsize=2048)
def predict_hourly(asset,origin,seed,values):
    import torch
    random.seed(seed);torch.manual_seed(seed)
    x=np.asarray(values,dtype=float);center=float(np.median(x))
    series=np.log(np.maximum(x,center*.001)/center)
    left=(-len(series))%32
    padded=np.pad(series,(left,0)).astype(np.float32)
    padding=np.r_[np.ones(left),np.zeros(len(series)+25)].astype(np.float32)
    model=pipeline()
    random.seed(seed);torch.manual_seed(seed)
    with torch.inference_mode():
        point,_=model.decode(torch.from_numpy(padded[None,:]),torch.from_numpy(padding[None,:]),torch.zeros((1,1),dtype=torch.long),horizon_len=25)
    point=point.detach().cpu().numpy()[0]
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
