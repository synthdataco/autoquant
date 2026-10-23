"""Fresh-process, uncached1000-path CPU latency and repeatability gate."""
import os,time,json,resource
started=time.perf_counter();os.environ['TORCH_DISABLE_NATIVE_JIT']='1'
import torch,numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from predict import load_minute_prices,context_end
from candidate_v9_research import simulate
from timesfm_vol import predict_hourly
framework=time.perf_counter()-started
t=pd.Timestamp('2026-10-08T12:17:00Z');end=context_end(t,3600)
frame=load_minute_prices(default_store_root('BTC').parents[2],'BTC',t-pd.Timedelta(days=8),t)
frame=frame.loc[(frame.index>end-pd.Timedelta(days=7))&(frame.index<=end)]
os.environ['MODEL_CONFIG']='{"timesfm":true}'
latencies=[];previous=None
for _ in range(3):
 predict_hourly.cache_clear();started=time.perf_counter();result=simulate('BTC',t,60,3600,1000,frame);latencies.append(time.perf_counter()-started)
 assert len(result)==1002 and np.isfinite(np.asarray(result[2:])).all()
 if previous is not None:assert previous==result
 previous=result
summary={'device':'cpu','framework_startup_seconds':framework,'uncached_simulate_seconds':latencies,'cold_response_seconds':framework+latencies[0],'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'paths':1000,'repeatable':True}
print(json.dumps(summary),flush=True)
assert summary['cold_response_seconds']<60 and max(latencies[1:])<5
