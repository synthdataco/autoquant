"""Single-GPU capacity probe; queued behind the existing model job."""
import time,json,os
from pathlib import Path
import numpy as np
from chronos_vol import pipeline,predict_hourly
import torch
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'));torch.manual_seed(seed)
values=tuple(np.exp(np.sin(np.arange(144)/12))*1e-6)
t=time.perf_counter();pipeline('chronos-t5-small');load=time.perf_counter()-t
latencies=[];forecasts=[]
for i in range(3):
 predict_hourly.cache_clear();t=time.perf_counter()
 forecasts.append(predict_hourly('TEST','origin',seed,values,'chronos-t5-small'))
 torch.cuda.synchronize();latencies.append(time.perf_counter()-t)
assert all(np.array_equal(forecasts[0],f) for f in forecasts[1:])
result={'load_seconds':load,'predict_seconds':latencies,'first_response_seconds':load+latencies[0],'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'repeatable':True}
Path('agent/results/chronos_small_latency.json').write_text(json.dumps(result,indent=2));print(result)
if load+latencies[0]>5 or max(latencies)>5:raise SystemExit('Preregistered practical latency gate failed')
