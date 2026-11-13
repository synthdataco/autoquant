"""Chronological, purged volatility regressions. One bounded GPU job at a time."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

from ml_features import NAMES, CORE_NAMES, TIME_NAMES, ASSETS
from run_suite import ROOT


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--dates',nargs='+',default=['2026-03-01','2026-07-01','2026-08-01','2026-10-01','2026-11-01','2026-11-09'])
    ap.add_argument('--rounds',type=int,default=600)
    ap.add_argument('--prefix',default='ml')
    ap.add_argument('--output',default='ml')
    ap.add_argument('--kinds',nargs='+',default=['vol','full'])
    ap.add_argument('--lookback',type=int,default=240)
    ap.add_argument('--min-child-weight',type=float,default=80)
    ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--validation-days',type=int,default=14)
    ap.add_argument('--scale-weight',action='store_true')
    ap.add_argument('--loss',choices=['log_mse','gamma'],default='log_mse')
    ap.add_argument('--horizons',nargs='+',type=int,default=[3600,86400])
    ap.add_argument('--assets',nargs='+')
    args=ap.parse_args()
    os.chdir(ROOT)
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261108'))
    frames=[pd.read_parquet(p) for p in sorted((ROOT/'agent/cache').glob(args.prefix+'_*.parquet'))]
    if len(frames)!=18:
        raise RuntimeError(f'Expected 18 feature tables, found {len(frames)}; finish make_ml_data first')
    data=pd.concat(frames,ignore_index=True).sort_values('start_time').reset_index(drop=True)
    if args.assets:
        data=data[data.asset.isin(args.assets)]
    out=ROOT/'agent/artifacts'/args.output
    out.mkdir(parents=True,exist_ok=True)
    manifest={'features':NAMES,'fits':[],'target':'log forward absolute-return scale / causal seasonal EWMA scale', 'xgboost_version':xgb.__version__,'cache_prefix':args.prefix}
    manifest['training_options']={'lookback_days':args.lookback,'min_child_weight':args.min_child_weight,'depth':args.depth,'validation_days':args.validation_days,'scale_weight':args.scale_weight}
    manifest['training_options']['loss']=args.loss
    manifest['training_options']['assets']=args.assets
    manifest['prediction_scale']='ratio' if args.loss=='gamma' else 'log_ratio'
    if args.loss=='gamma':
        manifest['target']='forward absolute-return scale / causal seasonal EWMA scale'
    features={'vol':CORE_NAMES+TIME_NAMES+['asset_'+a for a in ASSETS], 'full':NAMES}
    for date in args.dates:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            print('Landing order: stop fitting.',flush=True)
            break
        cutoff=pd.Timestamp(date,tz='UTC')
        for horizon in args.horizons:
            full=data[(data.time_length==horizon)&(data.label_end<cutoff)&(data.start_time>=cutoff-pd.Timedelta(days=args.lookback))].copy()
            split=cutoff-pd.Timedelta(days=args.validation_days)
            train=full[full.label_end<split]
            valid=full[full.start_time>=split]
            if len(train)<1000 or len(valid)<100:
                continue
            for kind,names in features.items():
                if kind not in args.kinds:
                    continue
                def matrix(frame):
                    x=frame[names].to_numpy(dtype=np.float32,copy=True)
                    x[~np.isfinite(x)]=np.nan
                    ratio=np.clip(frame.target_ratio.to_numpy(),.03,30)
                    y=ratio if args.loss=='gamma' else np.log(ratio)
                    weights=None
                    if args.scale_weight:
                        reference=max(float(train.base_sigma.median()),1e-8)
                        weights=np.clip(frame.base_sigma.to_numpy()/reference,.25,4.)
                        weights/=weights.mean()
                    return xgb.DMatrix(x,label=y,weight=weights,feature_names=names,nthread=2)
                dtrain,dvalid=matrix(train),matrix(valid)
                params={'objective':'reg:squarederror','tree_method':'hist','device':args.device,
                        'max_depth':args.depth,'eta':.035,'min_child_weight':args.min_child_weight,'lambda':15.,'subsample':.85,
                        'colsample_bytree':1.,'max_bin':128,'seed':seed,'nthread':2,'eval_metric':'rmse'}
                if args.loss=='gamma':
                    params.update({'objective':'reg:gamma','eval_metric':'gamma-deviance'})
                probe=xgb.train(params,dtrain,num_boost_round=args.rounds,evals=[(dvalid,'past_validation')],early_stopping_rounds=60,verbose_eval=False)
                rounds=int(probe.best_iteration)+1
                model=xgb.train(params,matrix(full),num_boost_round=rounds,verbose_eval=False)
                # CPU inference keeps GPU use confined to this fitting process.
                model.set_param({'device':'cpu','nthread':1})
                name=f'{kind}_{horizon}_{date}.ubj'
                model.save_model(out/name)
                item={'cutoff':date,'time_length':horizon,'kind':kind,'features':names,'file':name,
                      'training_rows':len(full),'purged_training_rows':len(train),'validation_rows':len(valid),
                      'rounds':rounds,'past_validation_metric':params['eval_metric'],'past_validation_score':float(probe.best_score),
                      'importance':model.get_score(importance_type='gain')}
                manifest['fits'].append(item)
                (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(date,horizon,kind,'rows',len(full),'rounds',rounds,'past',params['eval_metric'],probe.best_score,flush=True)
    print('Saved',len(manifest['fits']),'causal volatility models',flush=True)

if __name__=='__main__':
    main()
