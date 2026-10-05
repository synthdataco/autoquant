from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os
jobs=[('aug1','2026-08-01','2026-08-08',6),('aug2','2026-08-17','2026-08-24',6),('augmid','2026-08-08','2026-08-17',3)]
def run(job):
 w,start,end,hours=job;tag='long_'+w
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return 'landing'
 env=os.environ.copy();env.update(MODEL_MODE='har',MODEL_MIX='.4',SEASON_FILE='seasonality_abs48.json',HAR_FILE='har_long.json',OPENBLAS_NUM_THREADS='1')
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model','agent/research_modeling.py','--tag',tag,'--start',start,'--end',end,'--hours',str(hours),'--assets','BTC,ETH,SOL,XRP,HYPE'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
with ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(run,jobs):print(r,flush=True)
