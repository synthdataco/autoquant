"""Four bounded CPU workers for exact-origin official CRPS label collection."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,os,argparse
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['diagnostic','historical'],default='diagnostic');a=p.parse_args()

def run(asset):
    name='crps_slope_'+('historical_' if a.phase=='historical' else '')+asset
    with Path(f'agent/results/{name}.log').open('w') as log:
        result=subprocess.run(['uv','run','agent/collect_crps_slopes.py','--asset',asset,'--phase',a.phase],stdout=log,stderr=subprocess.STDOUT,
                              env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    print(asset,result.returncode,flush=True)
    if result.returncode:raise RuntimeError(asset)

with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,['BTC','ETH','SOL','XRP','HYPE']))
