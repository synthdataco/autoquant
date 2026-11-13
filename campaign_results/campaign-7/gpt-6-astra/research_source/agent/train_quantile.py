"""Purged conditional quantiles of future volatility, trained one GPU job at a time."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from ml_features import CORE_NAMES,TIME_NAMES,ASSETS
from run_suite import ROOT


def main():
    os.chdir(ROOT)
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    alpha=np.array([.05,.15,.3,.5,.7,.85,.95])
    names=CORE_NAMES+TIME_NAMES+['asset_'+a for a in ASSETS]
    data=pd.concat([pd.read_parquet(p) for p in sorted((ROOT/'agent/cache').glob('ml_*.parquet'))],ignore_index=True).sort_values('start_time')
    out=ROOT/'agent/artifacts/quantile'
    out.mkdir(parents=True,exist_ok=True)
    manifest={'quantiles':alpha.tolist(),'features':names,'fits':[]}
    for date in ['2026-03-01','2026-07-01','2026-08-01','2026-10-01','2026-11-01','2026-11-09']:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            break
        cutoff=pd.Timestamp(date,tz='UTC')
        for horizon in [3600,86400]:
            full=data[(data.time_length==horizon)&(data.label_end<cutoff)&(data.start_time>=cutoff-pd.Timedelta(days=240))]
            split=cutoff-pd.Timedelta(days=14)
            train=full[full.label_end<split]
            valid=full[full.start_time>=split]
            def matrix(frame):
                x=frame[names].to_numpy(dtype=np.float32,copy=True)
                x[~np.isfinite(x)]=np.nan
                return xgb.DMatrix(x,label=np.log(np.clip(frame.target_ratio,.03,30)),feature_names=names,nthread=2)
            params={'objective':'reg:quantileerror','quantile_alpha':alpha,'tree_method':'hist','device':'cuda',
                    'max_depth':3,'eta':.035,'min_child_weight':80,'lambda':15.,'subsample':.85,'max_bin':128,
                    'seed':seed,'nthread':2}
            probe=xgb.train(params,matrix(train),num_boost_round=600,evals=[(matrix(valid),'past_validation')],early_stopping_rounds=60,verbose_eval=False)
            rounds=int(probe.best_iteration)+1
            model=xgb.train(params,matrix(full),num_boost_round=rounds,verbose_eval=False)
            model.set_param({'device':'cpu','nthread':1})
            filename=f'quantile_{horizon}_{date}.ubj'
            model.save_model(out/filename)
            manifest['fits'].append({'cutoff':date,'time_length':horizon,'file':filename,'training_rows':len(full),'rounds':rounds,'past_quantile_loss':float(probe.best_score)})
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(date,horizon,'rows',len(full),'rounds',rounds,'past quantile loss',probe.best_score,flush=True)
    print('Saved',len(manifest['fits']),'conditional quantile models',flush=True)

if __name__=='__main__':
    main()
