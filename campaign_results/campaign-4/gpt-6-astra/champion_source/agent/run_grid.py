import concurrent.futures
import os
from pathlib import Path
import subprocess

jobs=[]
variants={
'constant':{'MODEL_MODE':'constant'},
'seasonal':{'MODEL_MODE':'seasonal'},
'adaptive':{'MODEL_MODE':'adaptive'},
'fast':{'MODEL_MODE':'adaptive_fast'},
'adaptive085':{'MODEL_MODE':'adaptive','MODEL_SCALE':'.85'},
'adaptive_t5':{'MODEL_MODE':'adaptive','MODEL_TAIL':'5'},
'adaptive_mix4':{'MODEL_MODE':'adaptive','MODEL_MIX':'.4'},
}
for window,start,end in [('aug1','2026-08-01','2026-08-08'),('aug2','2026-08-17','2026-08-24')]:
 for name,env in variants.items():
  tag=name+'_'+window
  # Already running/completed initial jobs.
  if tag in ['seasonal_aug1','adaptive_aug1']:continue
  jobs.append((tag,start,end,env))
def run(job):
 tag,start,end,extra=job
 budget=Path('agent/BUDGET.md').read_text()
 if 'LANDING ORDER' in budget:return tag,'landing'
 env=os.environ.copy();env.update(extra);env['OPENBLAS_NUM_THREADS']='1'
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model','agent/research_modeling.py','--tag',tag,'--start',start,'--end',end],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(run,jobs):print(result,flush=True)
