from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os,json
from synth.validator.competition_config import ALL_COMPETITIONS
from synth_lib.backtester.config import slug_for
jobs=[]
for c in ALL_COMPETITIONS:
 slug=slug_for(c)
 if not Path(f'offline_scores/rewards_history_{slug}.parquet').exists():continue
 for a in c.asset_list:
  if not Path(f'offline_scores/miner_scores_{a}_{slug}.parquet').exists():continue
  for model in ['baseline','candidate']:
   tag=model+'_nine'
   if Path(f'agent/results/{tag}_full_{a}_{c.time_length}.json').exists():continue
   jobs.append((tag,model,a,c.time_length))
def run(job):
 tag,name,asset,h=job
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return job,'landing'
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MODEL_MODE='har',MODEL_MIX='.4',SEASON_FILE='aug31_seasonality_abs48.json',HAR_FILE='aug31_har_abs48.json')
 path='agent/baseline.py' if name=='baseline' else 'agent/modeling.py'
 with open(f'agent/full_{tag}_{asset}_{h}.log','w') as log:
  p=subprocess.run(['uv','run','agent/full_backtest.py','--model',path,'--asset',asset,'--horizon',str(h),'--tag',tag,'--days','9'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return job,p.returncode
print('Jobs',len(jobs),flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(run,jobs):print(r,flush=True)
