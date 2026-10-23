"""Matched August score subsets and four-worker official heldout evaluation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os,json,argparse
import pandas as pd

assets=['BTC','ETH','SOL','XRP','HYPE']
p=argparse.ArgumentParser();p.add_argument('--artifact',default='crps_gradient_diagnostic.json');p.add_argument('--tag-prefix',default='crps_gradient');p.add_argument('--workers',type=int,default=4);p.add_argument('--kinds',nargs='+',choices=['tree','ridge'],default=['tree','ridge']);a=p.parse_args()
control=Path('agent/results/crps_gradient_control_holdout');control.mkdir(exist_ok=True)
for asset in assets:
    frame=pd.read_parquet(f'agent/results/crps_slope_labels/{asset}_base.parquet')
    frame=frame[frame.start_time>=pd.Timestamp('2026-08-01',tz='UTC')]
    prefix=control/f'{asset}_1h_2026-09-01'
    frame.to_parquet(prefix.with_suffix('.parquet'),index=False)
    prefix.with_suffix('.json').write_text(json.dumps({'num_prompts':len(frame),'mean_crps':float(frame.crps.mean()),
        'realized_coverage':1.,'mode':'subset_of_existing_official_baseline_scores'},indent=2))

def run(cell):
    asset,kind=cell
    with Path(f'agent/results/{a.tag_prefix}_{kind}_{asset}.log').open('w') as log:
        result=subprocess.run(['uv','run','agent/evaluate_crps_gradient.py','--asset',asset,'--kind',kind,'--artifact',a.artifact,'--tag-prefix',a.tag_prefix],stdout=log,stderr=subprocess.STDOUT,
                              env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    print(asset,kind,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(str(cell))

with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,[(asset,kind) for kind in a.kinds for asset in assets]))
