from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os

def run(scale):
 tag='fomc_'+scale.replace('.','p')
 env=os.environ.copy();env.update(MODEL_MODE='har',MODEL_MIX='.4',MODEL_FOMC=scale,SEASON_FILE='jul20_seasonality_abs48.json',HAR_FILE='jul20_har_abs48.json',OPENBLAS_NUM_THREADS='1')
 if 'LANDING ORDER' in Path('agent/BUDGET.md').read_text():return 'landing'
 with open('agent/results_'+tag+'.log','w') as log:
  p=subprocess.run(['uv','run','agent/predict.py','--research','--model','agent/research_modeling.py','--tag',tag,'--start','2026-07-28','--end','2026-07-31','--hours','1','--assets','BTC,ETH,SOL,XRP,HYPE'],env=env,stdout=log,stderr=subprocess.STDOUT)
 return tag,p.returncode
with ThreadPoolExecutor(max_workers=3) as pool:
 for r in pool.map(run,['1','1.5','2']):print(r,flush=True)
