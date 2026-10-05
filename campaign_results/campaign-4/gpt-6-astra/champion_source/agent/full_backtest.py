"""Full official orchestration, using only locally cached allowed historical scores."""
import argparse,importlib.util,json,os
from filelock import FileLock
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from predict import ensure_workspace_cwd,load_lenient,check_snapshot_coverage
from synth_lib.backtester.orchestration import backtest
from synth_lib.backtester.loading import get_miner_scores
from synth_lib.benchmark.generate_predictions import context_end

def main():
 ensure_workspace_cwd()
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--horizon',type=int,default=86400);ap.add_argument('--tag',required=True);ap.add_argument('--days',type=int,default=9);ap.add_argument('--end',default='2026-09-13');args=ap.parse_args()
 os.environ['SYNTH_BACKTESTER_OFFLINE_DATA_ROOT']=str(Path('offline_scores').resolve())
 spec=importlib.util.spec_from_file_location('candidate',args.model);model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
 end=pd.Timestamp(args.end,tz='UTC');start=end-pd.Timedelta(days=args.days);h=args.horizon;step=60 if h==3600 else 300
 scores=get_miner_scores(start.to_pydatetime(),end.to_pydatetime(),args.asset,h,step)
 times=pd.DatetimeIndex(sorted(scores.start_time.unique()))
 prices=load_lenient(args.asset,times.min()-pd.Timedelta(days=8),times.max()+pd.Timedelta(hours=26))
 out=Path('miner_outputs/full')/args.tag/args.asset/f'{h}_{step}';out.mkdir(parents=True,exist_ok=True)
 for day in times.normalize().unique():
  daytimes=times[times.normalize()==day];fans=[]
  for t in daytimes:
   ctx=prices.loc[t-pd.Timedelta(days=7):context_end(t,h)]
   paths=np.asarray(model.simulate(args.asset,t.isoformat(),step,h,1000,ctx)[2:],dtype=np.float32)
   assert paths.shape==(1000,h//step+1) and np.isfinite(paths).all()
   fans.append(paths)
  target=out/f'date={day.date()}.npy';np.save(target,np.stack(fans))
  target.with_suffix('.json').write_text(json.dumps(dict(asset=args.asset,time_length=h,time_increment=step,num_simulations=1000,num_steps=h//step+1,start_times=[t.isoformat() for t in daytimes])))
 print('Generated exact scored prompts:',len(times),flush=True)
 # All possible price days were checked by load_lenient, and offline mode prevents
 # realized-path fetching. No market data ingestion can be needed in this range.
 with ProcessPoolExecutor(max_workers=4) as pool:
  result=backtest(miner_name=args.tag,asset=args.asset,time_length=h,time_increment=step,n_backtest_days=args.days,predictions_dir=out.parents[1],eval_end=end.to_pydatetime(),scoring_executor=pool)
 p=Path('agent/results')/(args.tag+'_full_'+args.asset+'_'+str(h))
 p.with_suffix('.json').write_text(json.dumps(result.summary,indent=2,default=str))
 result.prompt_df.loc[result.prompt_df.miner_uid==999].to_csv(p.with_suffix('.csv'),index=False)
 print(result.summary,flush=True)
 for f in out.glob('date=*.npy'):f.unlink()
if __name__=='__main__':
 import sys
 asset=sys.argv[sys.argv.index('--asset')+1]
 tag=sys.argv[sys.argv.index('--tag')+1]
 horizon=sys.argv[sys.argv.index('--horizon')+1] if '--horizon' in sys.argv else '86400'
 lockroot=Path('miner_outputs/full/locks');lockroot.mkdir(parents=True,exist_ok=True)
 with FileLock(str(lockroot/f'{tag}_{asset}_{horizon}.lock')):
  if not Path(f'agent/results/{tag}_full_{asset}_{horizon}.json').exists():main()
