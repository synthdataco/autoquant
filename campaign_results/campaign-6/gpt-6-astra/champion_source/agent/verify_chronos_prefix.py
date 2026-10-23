"""Verify short decoding preserves the seeded forecast prefix exactly."""
import os,random,time,json
import numpy as np,pandas as pd,torch
from synth_lib.preparation.config import default_store_root
from predict import load_minute_prices,context_end
from research_model import shape_at
from chronos_vol import pipeline
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
model=pipeline();rows=[]
for asset,date in [('BTC','2026-09-16T00:02:00Z'),('HYPE','2026-10-08T13:17:00Z')]:
 t=pd.Timestamp(date);frame=load_minute_prices(default_store_root(asset).parents[2],asset,t-pd.Timedelta(days=8),t)
 end=context_end(t,3600);frame=frame.loc[(frame.index>end-pd.Timedelta(days=7))&(frame.index<=end)]
 ret=np.log(frame.close.astype(float)).diff();rv=ret.pow(2)/shape_at(frame.index,asset,True)
 hourly=rv.resample('h').mean();count=frame.close.resample('h').size()
 hourly=hourly[(count==60)&(hourly.index+pd.Timedelta(hours=1)<=frame.index[-1]+pd.Timedelta(minutes=1))].iloc[-144:]
 values=hourly.to_numpy();center=float(np.nanmedian(values));series=np.log(np.maximum(values,center*.001)/center)
 forecasts={};elapsed={}
 for length in [25,3]:
  random.seed(seed);torch.manual_seed(seed)
  if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
  started=time.perf_counter()
  with torch.inference_mode():pred=model.predict(torch.tensor(series,dtype=torch.float32),prediction_length=length,num_samples=32)
  forecasts[length]=pred.detach().cpu().numpy()[0];elapsed[length]=time.perf_counter()-started
 mean25=np.mean(np.exp(np.clip(forecasts[25],-3,3)),axis=0)*center
 mean3=np.mean(np.exp(np.clip(forecasts[3],-3,3)),axis=0)*center
 origin=hourly.index[-1]+pd.Timedelta(hours=1)
 max_index=int((t+pd.Timedelta(hours=1)-origin).total_seconds()//3600)
 row={'asset':asset,'start_time':str(t),'sample_prefix_equal':bool(np.array_equal(forecasts[25][:,:3],forecasts[3])),'mean_prefix_equal':bool(np.array_equal(mean25[:3],mean3)),'max_mean_difference':float(np.max(np.abs(mean25[:3]-mean3))),'required_forecast_values':max_index+1,'seconds':elapsed}
 rows.append(row);print(json.dumps(row),flush=True)
assert all(r['sample_prefix_equal'] and r['mean_prefix_equal'] and r['required_forecast_values']<=3 for r in rows)
