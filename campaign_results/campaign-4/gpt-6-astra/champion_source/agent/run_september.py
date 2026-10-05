from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os
jobs=[('baseline_september','agent/baseline.py',{}),('julyfit_september','agent/research_modeling.py',dict(MODEL_MODE='har',MODEL_MIX='.4',SEASON_FILE='seasonality_abs48.json',HAR_FILE='har_abs48.json')),('augfit_september','agent/research_modeling.py',dict(MODEL_MODE='har',MODEL_MIX='.4',SEASON_FILE='aug31_seasonality_abs48.json',HAR_FILE='aug31_har_abs48.json'))]
def run(job):
 tag,model,extra=job
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return 'landing'
 env=os.environ.copy();env.update(extra);env['OPENBLAS_NUM_THREADS']='1'
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model',model,'--tag',tag,'--start','2026-09-04','--end','2026-09-14','--hours','3'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
if __name__=='__main__':
 # The caller launches this after the fresh HAR artifact has finished fitting.
 with ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(run,jobs):print(r,flush=True)
