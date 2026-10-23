"""Bounded parallel official-backtester experiments, with durable per-cell artifacts."""
import argparse, subprocess, os, json
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CRYPTO = ['BTC','ETH','SOL','XRP','HYPE']
COM = ['XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']
p=argparse.ArgumentParser();p.add_argument('--forecast-cache');p.add_argument('--forecast-cache-only',action='store_true');p.add_argument('--score-details',action='store_true');p.add_argument('--skip-crypto24h',action='store_true');p.add_argument('--prediction-dtype',choices=['float32','float64'],default='float64');p.add_argument('--cached-prompts',action='store_true');p.add_argument('--research',action='store_true');p.add_argument('--seed',type=int,default=20261011);p.add_argument('--config',default='{}');p.add_argument('--tag',required=True);p.add_argument('--model',default='modeling');p.add_argument('--ends',nargs='+',default=['2026-09-16','2026-10-10']);p.add_argument('--days',type=int,default=2);p.add_argument('--workers',type=int,default=4);p.add_argument('--assets',nargs='+',default=CRYPTO+COM);p.add_argument('--formats',nargs='+',default=['24h','1h']);p.add_argument('--paths',type=int,default=1000)
a=p.parse_args();
if a.forecast_cache_only and not a.forecast_cache:p.error('--forecast-cache-only requires --forecast-cache')
root=Path('agent/results')/a.tag;root.mkdir(parents=True,exist_ok=True)
(root/'config.txt').write_text(str(vars(a)))
def run(cell):
 asset,fmt,end=cell;prefix=root/f'{asset}_{fmt}_{end}'
 if prefix.with_suffix('.json').exists() and prefix.with_suffix('.parquet').exists():
  try:
   summary=json.loads(prefix.with_suffix('.json').read_text())
   if summary['num_prompts']>0 and pq.ParquetFile(prefix.with_suffix('.parquet')).metadata.num_rows>0:return
  except (ValueError,OSError,KeyError):pass
 cmd=['uv','run','agent/predict.py','--asset',asset,'--model',a.model,'--days',str(a.days),'--eval-end',end,'--num-simulations',str(a.paths),'--latest-formula','--output',str(prefix),'--prediction-dtype',a.prediction_dtype]
 if a.research:cmd[2:2]=['--group','research']
 if a.cached_prompts:cmd+=['--cached-prompts']
 if a.score_details:cmd+=['--score-details']
 if fmt=='1h':cmd+=['--time-length','3600','--time-increment','60']
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MODEL_CONFIG=a.config,SYNTH_BENCHMARK_SEED=str(a.seed))
 if a.forecast_cache:env['CHRONOS_RESEARCH_CACHE']=a.forecast_cache
 if a.forecast_cache_only:env['CHRONOS_RESEARCH_CACHE_ONLY']='1'
 with prefix.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 print(prefix,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(str(prefix))
cells=[(asset,fmt,end) for end in a.ends for asset in a.assets for fmt in a.formats if (fmt=='24h' or asset in CRYPTO) and not(a.skip_crypto24h and asset in CRYPTO and fmt=='24h')]
with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,cells))
