from concurrent.futures import ThreadPoolExecutor
import os,subprocess
from pathlib import Path
jobs=[]
variants={
'har_orig':{'MODEL_MODE':'har','MODEL_MIX':'.4'},
'har48':{'MODEL_MODE':'har','MODEL_MIX':'.4','SEASON_FILE':'seasonality_abs48.json','HAR_FILE':'har_abs48.json'},
'har48_09':{'MODEL_MODE':'har','MODEL_MIX':'.4','MODEL_SCALE':'.9','SEASON_FILE':'seasonality_abs48.json','HAR_FILE':'har_abs48.json'},
'har48_ma':{'MODEL_MODE':'har','MODEL_MIX':'.4','MODEL_DEPENDENCE':'fitted','SEASON_FILE':'seasonality_abs48.json','HAR_FILE':'har_abs48.json'},
}
for name,env in variants.items():jobs.append((name+'_augmid','2026-08-08','2026-08-17',3,env))
for seed in ['123','987']:
 for window,start,end in [('aug1','2026-08-01','2026-08-08'),('aug2','2026-08-17','2026-08-24')]:
  for name in ['har48','har48_ma']:
   jobs.append((name+'_'+window+'_s'+seed,start,end,6,{**variants[name],'SYNTH_BENCHMARK_SEED':seed}))
def run(job):
 tag,start,end,hours,extra=job
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return tag,'landing'
 env=os.environ.copy();env.update(extra);env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model','agent/research_modeling.py','--tag',tag,'--start',start,'--end',end,'--hours',str(hours)],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
with ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(run,jobs):print(result,flush=True)
