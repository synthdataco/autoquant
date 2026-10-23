"""Offline Chronos residual-volatility experiment; one GPU worker only."""
from functools import lru_cache
from pathlib import Path
import os,random,hashlib,json
import numpy as np
import pandas as pd

# PyTorch's optional native-JIT overrides require a system C compiler absent here.
# Its precompiled CUDA operators implement the same inference without compilation.
os.environ['TORCH_DISABLE_NATIVE_JIT']='1'

@lru_cache(maxsize=1)
def pipeline(model_name='chronos-t5-tiny'):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from chronos import ChronosPipeline
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.2)
    return ChronosPipeline.from_pretrained(str(Path(__file__).parent/'models'/model_name),device_map=device,dtype=torch.float32,local_files_only=True)

@lru_cache(maxsize=2048)
def predict_hourly(asset,origin,seed,values,model_name='chronos-t5-tiny'):
    cache_file=None
    cache_root=os.environ.get('CHRONOS_RESEARCH_CACHE')
    if cache_root:
        model_dir=Path(__file__).parent/'models'/model_name
        provenance_path=model_dir/'provenance.json'
        if not provenance_path.exists():provenance_path=model_dir/'revision.txt'
        provenance=provenance_path.read_bytes()
        key=hashlib.sha256(Path(__file__).read_bytes()+provenance+json.dumps([asset,origin,seed,model_name]).encode()+np.asarray(values,dtype=np.float64).tobytes()).hexdigest()
        cache_file=Path(cache_root)/key[:2]/(key+'.npy')
        try:
            saved=np.load(cache_file,allow_pickle=False)
            if saved.shape==(25,) and np.all(np.isfinite(saved)) and np.all(saved>0):return saved
        except (OSError,ValueError):pass
        if os.environ.get('CHRONOS_RESEARCH_CACHE_ONLY')=='1':
            raise FileNotFoundError('Research forecast cache miss: '+str(cache_file))
    import torch
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    x=np.asarray(values,dtype=float)
    center=float(np.nanmedian(x))
    if not np.isfinite(center) or center<=0:return np.full(25,1e-8)
    series=np.log(np.maximum(x,center*.001)/center)
    model=pipeline(model_name)
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        pred=model.predict(torch.tensor(series,dtype=torch.float32),prediction_length=25,num_samples=32)
    pred=pred.detach().cpu().numpy()[0]
    result=np.mean(np.exp(np.clip(pred,-3,3)),axis=0)*center
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True,exist_ok=True)
        temporary=cache_file.with_suffix('.'+str(os.getpid())+'.tmp')
        with temporary.open('wb') as stream:np.save(stream,result,allow_pickle=False)
        temporary.replace(cache_file)
    return result

def forecast_variance(frame,asset,future,seed,persistence=False,model_name='chronos-t5-tiny',shape_name='seasonality.json'):
    from research_model import shape_at
    ret=np.log(frame.close.astype(float)).diff()
    rv=ret.pow(2)/shape_at(frame.index,asset,True,shape_name)
    hourly=rv.resample('h').mean()
    count=frame.close.resample('h').size()
    # Complete clock hours only. Missing trades remain NaN; never interpolate prices.
    hourly=hourly[(count==60)&(hourly.index+pd.Timedelta(hours=1)<=frame.index[-1]+pd.Timedelta(minutes=1))].iloc[-144:]
    if hourly.notna().sum()<48:return np.full(len(future),float(rv.mean()))
    if persistence:
        if persistence=='median':estimate=hourly.median()
        elif persistence=='ewma':estimate=hourly.ewm(halflife=6).mean().iloc[-1]
        else:estimate=hourly.dropna().iloc[-1]
        return np.full(len(future),float(estimate))
    origin=hourly.index[-1]+pd.Timedelta(hours=1)
    values=tuple(hourly.to_numpy(dtype=float))
    forecast=predict_hourly(asset,str(origin),seed,values,model_name)
    at=np.asarray((future-origin).total_seconds())//3600
    return forecast[np.clip(at.astype(int),0,len(forecast)-1)]
