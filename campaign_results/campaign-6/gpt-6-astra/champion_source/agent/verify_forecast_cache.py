"""Exact opt-in disk-cache parity; run as the sole GPU process."""
import os,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from predict import load_minute_prices,context_end
import chronos_vol
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
t=pd.Timestamp('2026-09-16T12:17:00Z')
frame=load_minute_prices(default_store_root('BTC').parents[2],'BTC',t-pd.Timedelta(days=8),t)
end=context_end(t,3600);frame=frame.loc[(frame.index>end-pd.Timedelta(days=7))&(frame.index<=end)]
future=pd.date_range(t+pd.Timedelta(minutes=1),periods=60,freq='min')
os.environ.pop('CHRONOS_RESEARCH_CACHE',None)
a=chronos_vol.forecast_variance(frame,'BTC',future,seed)
Path('agent/results/chronos_forecast_cache').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir='agent/results/chronos_forecast_cache') as root:
 os.environ['CHRONOS_RESEARCH_CACHE']=root
 chronos_vol.predict_hourly.cache_clear()
 b=chronos_vol.forecast_variance(frame,'BTC',future,seed)
 assert np.array_equal(a,b)
 files=list(Path(root).rglob('*.npy'));assert len(files)==1
 chronos_vol.predict_hourly.cache_clear()
 original=chronos_vol.pipeline
 def forbidden(*args,**kwargs):raise AssertionError('Disk cache unexpectedly ran inference')
 chronos_vol.pipeline=forbidden
 c=chronos_vol.forecast_variance(frame,'BTC',future,seed)
 assert np.array_equal(a,c)
 os.environ['CHRONOS_RESEARCH_CACHE_ONLY']='1'
 try:chronos_vol.forecast_variance(frame,'BTC',future,seed+1)
 except FileNotFoundError:pass
 else:raise AssertionError('Different seed incorrectly reused the cached forecast')
 os.environ.pop('CHRONOS_RESEARCH_CACHE_ONLY',None)
 chronos_vol.pipeline=original
os.environ.pop('CHRONOS_RESEARCH_CACHE',None)
print(json.dumps({'exact_uncached_disk_parity':True,'cache_reload_without_inference':True,'forecast_points':len(a)}))
