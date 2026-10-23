"""Paired official loss labels via predict.py; snapshot-only, causal July profiles."""
import argparse,json,os,tempfile,shutil,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
import predict
from candidate_v9_research import simulate
from research_model import residual_features

p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--phase',choices=['diagnostic','historical'],default='diagnostic');a=p.parse_args()
predict.require_offline_bundle();predict.ensure_workspace_cwd()
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
historical=a.phase=='historical'
available='2026-04-01' if historical else '2026-07-01'
suffix='apr01' if historical else 'jul01'
source=pd.read_parquet('agent/results/nonlinear_'+a.phase+'_data.parquet')
source=source[(source.asset==a.asset)&(source.start_time>=pd.Timestamp(available,tz='UTC'))]
times=sorted(pd.to_datetime(source.start_time,utc=True).tolist())
predict.check_snapshot_coverage(a.asset,times[0]-pd.Timedelta(days=7),times[-1]+pd.Timedelta(hours=1))
prices=predict.load_minute_prices(default_store_root(a.asset).parents[2],a.asset,times[0]-pd.Timedelta(days=7),times[-1]+pd.Timedelta(hours=1))
kept=[]
for t in times:
    reference=prices.close.reindex(pd.date_range(t,periods=61,freq='min')).to_numpy(dtype=float)
    if np.all(np.isfinite(reference)) and np.all(reference>0):kept.append(t)
    assert predict.context_end(t,3600)+pd.Timedelta(minutes=1)>=pd.Timestamp(available,tz='UTC')
metadata={'asset':a.asset,'original_origins':len(times),'complete_reference_origins':len(kept),
          'excluded_missing_reference':len(times)-len(kept),'seed':seed,'paths':1000,'prediction_dtype':'float32',
          'source_sha256':hashlib.sha256(Path('agent/research_model.py').read_bytes()).hexdigest(),
          'training_profile_available':available,'label_end_exclusive':'2026-07-01' if historical else '2026-09-01'}
root=Path('agent/results/crps_slope_historical_labels' if historical else 'agent/results/crps_slope_labels');root.mkdir(exist_ok=True)
(root/(a.asset+'_manifest.json')).write_text(json.dumps(metadata,indent=2))
if historical:
    rows=[]
    for t in kept:
        context=prices.loc[t-pd.Timedelta(days=7):predict.context_end(t,3600)]
        row=residual_features(context,a.asset,'seasonality_'+suffix+'.json')
        row.update(asset=a.asset,start_time=t);rows.append(row)
    pd.DataFrame(rows).to_parquet(root/(a.asset+'_features.parquet'),index=False)
predict.simulate=simulate
predict.PREDICTIONS_ROOT.mkdir(parents=True,exist_ok=True)
for name,shift in [('minus',-.1),('base',0.),('plus',.1)]:
    prefix=root/(a.asset+'_'+name)
    config={'table':'seasonality_'+suffix+'.json','clock_table':'clock_'+suffix+'.json','event_table':'event_multipliers_'+suffix+'.json','scale':float(np.exp(shift))}
    if prefix.with_suffix('.parquet').exists() and prefix.with_suffix('.json').exists():
        saved=json.loads(prefix.with_suffix('.json').read_text())
        if saved['metadata']!=metadata or saved['config']!=config:
            raise ValueError('Existing loss labels have different provenance: '+str(prefix))
        if saved['summary']['num_prompts']!=len(kept):raise ValueError('Incomplete label artifact')
        continue
    os.environ['MODEL_CONFIG']=json.dumps(config)
    out=Path(tempfile.mkdtemp(prefix='crps-slope-',dir=predict.PREDICTIONS_ROOT))
    try:
        predict.generate(a.asset,kept,out,60,3600,1000,'float32')
        result=predict.cached_backtest(a.asset,kept,{},out,3600,60,True)
        assert len(result.prompt_df)==len(kept) and (result.prompt_df.crps>=0).all()
        temporary=prefix.with_suffix('.parquet.tmp');result.prompt_df.to_parquet(temporary,index=False);temporary.replace(prefix.with_suffix('.parquet'))
        prefix.with_suffix('.json').write_text(json.dumps({'summary':result.summary,'config':config,'metadata':metadata},default=str,indent=2))
        print(a.asset,name,result.summary['num_prompts'],result.summary['mean_crps'],flush=True)
    finally:shutil.rmtree(out)
