"""Snapshot-only evaluation through the official backtester scorer.
Invoked by predict.py --research. No CRPS implementation lives here.
"""
import argparse
import importlib.util
import json
import time
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from synth_lib.backtester.scoring import _score_single_prompt
from synth_lib.backtester.config import competition_for, VOL_CRPS_1H_DATE
from synth_lib.backtester.preparation import PredictionRef, _slice_real_prices
from synth_lib.preparation.realized_path_store import RealizedPathStore
from synth_lib.benchmark.generate_predictions import context_end


def main():
    from predict import load_lenient, ensure_workspace_cwd
    ensure_workspace_cwd()
    ap=argparse.ArgumentParser()
    ap.add_argument('--research',action='store_true')
    ap.add_argument('--model',default='agent/baseline.py')
    ap.add_argument('--assets',default='BTC,ETH,SOL,XRP,HYPE,XAU,NVDAX,GOOGLX,TSLAX,AAPLX,WTIOIL,SPCX,SP500')
    ap.add_argument('--start',default='2026-08-01')
    ap.add_argument('--end',default='2026-08-08')
    ap.add_argument('--hours',type=int,default=6)
    ap.add_argument('--paths',type=int,default=1000)
    ap.add_argument('--horizons',default='86400,3600')
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('candidate',args.model)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    start=pd.Timestamp(args.start,tz='UTC');end=pd.Timestamp(args.end,tz='UTC')
    assert end+pd.Timedelta(days=1)<=pd.Timestamp('2026-09-15',tz='UTC')
    out=Path('agent/results');out.mkdir(exist_ok=True)
    tmp=Path('miner_outputs/research')/args.tag;tmp.mkdir(parents=True,exist_ok=True)
    rows=[]
    for asset in args.assets.split(','):
        # end is exclusive; the last prompt cannot need the next midnight bar.
        prices=load_lenient(asset,start-pd.Timedelta(days=8),end+pd.Timedelta(days=1)-pd.Timedelta(minutes=1))
        for horizon in map(int,args.horizons.split(',')):
            if horizon==3600 and asset not in ['BTC','ETH','SOL','XRP','HYPE']:continue
            step=60 if horizon==3600 else 300
            comp=competition_for(asset,horizon)
            realized=RealizedPathStore(asset,horizon,step)
            for t in pd.date_range(start,end,freq=f'{args.hours}h',inclusive='left'):
                ctx=prices.loc[t-pd.Timedelta(days=7):context_end(t,horizon)]
                before=time.perf_counter()
                paths=np.asarray(model.simulate(asset,t.isoformat(),step,horizon,args.paths,ctx)[2:],dtype=np.float32)
                latency=time.perf_counter()-before
                assert paths.shape==(args.paths,horizon//step+1) and np.isfinite(paths).all() and (paths>0).all()
                np.save(tmp/'paths.npy',paths[None])
                (tmp/'paths.json').write_text(json.dumps(dict(asset=asset,time_length=horizon,time_increment=step,num_simulations=args.paths,num_steps=paths.shape[1],start_times=[t.isoformat()])))
                real=_slice_real_prices(prices[['close']],t,horizon,step)
                source='minutes'
                if not np.isfinite(real).all():
                    cached=realized.get(t)
                    if cached is not None and len(cached)==len(real):
                        real=cached.tolist();source='cached'
                ref=PredictionRef(tmp/'paths.npy',0)
                # Version replay: score current published rules on past observations.
                score=_score_single_prompt(ref,t,asset,VOL_CRPS_1H_DATE,horizon,step,real,comp.scoring_intervals,comp.vol_scoring_blocks,999)
                row=dict(asset=asset,horizon=horizon,start=t.isoformat(),crps=score['crps'],coverage=float(np.isfinite(real).mean()),source=source,latency=latency)
                if horizon==3600:
                    historical=_score_single_prompt(ref,t,asset,t+pd.Timedelta(seconds=horizon),horizon,step,real,comp.scoring_intervals,comp.vol_scoring_blocks,999)
                    row['historical_crps']=historical['crps']
                rows.append(row)
            df=pd.DataFrame(rows)
            df.to_csv(out/f'{args.tag}.csv',index=False)
            group=df[(df.asset==asset)&(df.horizon==horizon)]
            print(json.dumps(dict(tag=args.tag,asset=asset,horizon=horizon,n=len(group),crps=group.crps.mean(),coverage=group.coverage.mean(),latency=group.latency.mean())),flush=True)
    metadata=vars(args).copy()
    metadata['environment']={k:v for k,v in os.environ.items() if k.startswith('MODEL_') or k in ['SEASON_FILE','HAR_FILE','SYNTH_BENCHMARK_SEED']}
    metadata['model_sha256']=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    metadata['scoring']='synth_lib.backtester.scoring._score_single_prompt; current-rule replay, plus historical 1h'
    (out/f'{args.tag}.json').write_text(json.dumps(metadata,indent=2))
    for f in tmp.glob('paths.*'):f.unlink()

if __name__=='__main__':main()
