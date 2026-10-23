"""Exact-origin heldout official CRPS, using the same complete snapshot references."""
import argparse,json,os,tempfile,shutil,hashlib
from pathlib import Path
import pandas as pd
import predict
from candidate_v9_research import simulate

p=argparse.ArgumentParser();p.add_argument('--artifact',default='crps_gradient_diagnostic.json');p.add_argument('--tag-prefix',default='crps_gradient');p.add_argument('--asset',required=True);p.add_argument('--kind',choices=['tree','ridge'],required=True);a=p.parse_args()
predict.require_offline_bundle();predict.ensure_workspace_cwd();predict.simulate=simulate
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
manifest=json.loads(Path(f'agent/results/crps_slope_labels/{a.asset}_manifest.json').read_text())
assert seed==manifest['seed']
baseline=pd.read_parquet(f'agent/results/crps_slope_labels/{a.asset}_base.parquet')
baseline=baseline[baseline.start_time>=pd.Timestamp('2026-08-01',tz='UTC')]
times=sorted(pd.to_datetime(baseline.start_time,utc=True).tolist())
predict.check_snapshot_coverage(a.asset,times[0]-pd.Timedelta(days=7),times[-1]+pd.Timedelta(hours=1))
config={'table':'seasonality_jul01.json','clock_table':'clock_jul01.json','event_table':'event_multipliers_jul01.json',
        'nonlinear':a.kind,'nonlinear_model':a.artifact}
os.environ['MODEL_CONFIG']=json.dumps(config)
root=Path('agent/results/'+a.tag_prefix+'_'+a.kind+'_holdout');root.mkdir(exist_ok=True)
prefix=root/f'{a.asset}_1h_2026-09-01'
model_sha=hashlib.sha256((Path('agent')/a.artifact).read_bytes()).hexdigest()
if prefix.with_suffix('.json').exists() and prefix.with_suffix('.parquet').exists():
    saved=json.loads(prefix.with_suffix('.json').read_text())
    assert saved['model_sha256']==model_sha and saved['benchmark_seed']==seed
    if saved['num_prompts']==len(times):raise SystemExit(0)
predict.PREDICTIONS_ROOT.mkdir(parents=True,exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='gradient-holdout-',dir=predict.PREDICTIONS_ROOT))
try:
    predict.generate(a.asset,times,out,60,3600,1000,'float32')
    result=predict.cached_backtest(a.asset,times,{},out,3600,60,True)
    result.summary.update(model_sha256=model_sha,benchmark_seed=seed)
    assert len(result.prompt_df)==len(times) and (result.prompt_df.crps>=0).all()
    result.prompt_df.to_parquet(prefix.with_suffix('.parquet'),index=False)
    prefix.with_suffix('.json').write_text(json.dumps(result.summary,default=str,indent=2))
    print(a.asset,a.kind,len(times),result.summary['mean_crps'],flush=True)
finally:shutil.rmtree(out)
