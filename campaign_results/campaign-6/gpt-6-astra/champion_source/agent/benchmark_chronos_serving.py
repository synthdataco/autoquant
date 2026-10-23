"""Identical fresh-process serving benchmarks for public Chronos checkpoints."""
import argparse,time,os,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',choices=['chronos-t5-tiny','chronos-t5-small'],required=True);a=p.parse_args()
begin=time.perf_counter()
from chronos_vol import pipeline,predict_hourly
import numpy as np,pandas as pd,torch
from chronos import ChronosPipeline
import research_model
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'));torch.manual_seed(seed);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
framework=time.perf_counter()-begin
before=time.perf_counter();pipeline(a.model);load=time.perf_counter()-before
# Serving receives context from its caller; snapshot I/O is excluded from model latency.
from predict import load_minute_prices,WORKSPACE,context_end
t=pd.Timestamp('2026-09-15T12:00:00Z')
frame=load_minute_prices(WORKSPACE/'market_data','BTC',t-pd.Timedelta(days=7),t).loc[:context_end(t,3600)]
os.environ['MODEL_CONFIG']=json.dumps({'seasonal':True,'fast_half':15,'chronos':True,'chronos_model':a.model})
latencies=[];first=None
for _ in range(3):
 predict_hourly.cache_clear();before=time.perf_counter()
 out=research_model.simulate('BTC',t.isoformat(),60,3600,1000,frame)
 torch.cuda.synchronize();latencies.append(time.perf_counter()-before)
 paths=np.asarray(out[2:]);assert paths.shape==(1000,61) and np.isfinite(paths).all()
 if first is None:first=paths
 else:assert np.array_equal(first,paths)
result={'model':a.model,'framework_startup_seconds':framework,'model_load_seconds':load,'simulate_seconds':latencies,'cold_response_seconds':framework+load+latencies[0],'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'paths':1000,'repeatable':True}
Path(f'agent/results/{a.model}_serving_latency.json').write_text(json.dumps(result,indent=2));print(result)
if result['cold_response_seconds']>=60 or max(latencies[1:])>=5:raise SystemExit('Revised latency gate failed')
