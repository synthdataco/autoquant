from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os
jobs=[(scale,mix) for scale in [.8,.9,1.,1.1,1.2] for mix in [0.,.2,.4,.6]]
def run(job):
 scale,mix=job;tag=f'cal_s{scale:.1f}_m{mix:.1f}'.replace('.','p')
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return tag,'landing'
 env=os.environ.copy();env.update(MODEL_MODE='har',MODEL_SCALE=str(scale),MODEL_MIX=str(mix),SEASON_FILE='jul20_seasonality_abs48.json',HAR_FILE='jul20_har_abs48.json',OPENBLAS_NUM_THREADS='1')
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model','agent/research_modeling.py','--tag',tag,'--start','2026-07-21','--end','2026-07-28','--hours','6'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(run,jobs):print(r,flush=True)
