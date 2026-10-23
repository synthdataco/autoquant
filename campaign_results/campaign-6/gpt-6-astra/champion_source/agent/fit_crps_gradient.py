"""Learn a conditional official-loss slope; actual heldout CRPS is the gate."""
import json,os,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

p=argparse.ArgumentParser();p.add_argument('--target',choices=['gradient','proxy'],default='gradient');p.add_argument('--phase',choices=['diagnostic','refit','historical'],default='diagnostic');a=p.parse_args()
if a.target=='proxy' and a.phase=='historical':p.error('Historical proxy requires separately reconstructed causal targets')
stem='crps_gradient' if a.target=='gradient' else 'matched_proxy'
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
historical=a.phase=='historical'
root=Path('agent/results/crps_slope_historical_labels' if historical else 'agent/results/crps_slope_labels')
features=['fast_ratio','slow_ratio','range_ratio','volume_ratio','trade_ratio','return_signed']
source=pd.concat([pd.read_parquet(root/(asset+'_features.parquet')) for asset in ['BTC','ETH','SOL','XRP','HYPE']],ignore_index=True) if historical else pd.read_parquet('agent/results/nonlinear_diagnostic_data.parquet')
parts=[]
for asset in ['BTC','ETH','SOL','XRP','HYPE']:
    manifest=json.loads((root/(asset+'_manifest.json')).read_text())
    assert manifest['seed']==seed and manifest['paths']==1000
    scores=None
    for name in ['minus','base','plus']:
        frame=pd.read_parquet(root/(asset+'_'+name+'.parquet'))[['asset','start_time','crps']].rename(columns={'crps':name})
        if scores is None:scores=frame
        else:
            old=len(scores);scores=scores.merge(frame,on=['asset','start_time'],validate='one_to_one')
            assert len(scores)==old==len(frame)
    columns=['asset','start_time']+features+(['target'] if a.target=='proxy' else [])
    joined=scores.merge(source[columns].rename(columns={'target':'proxy_target'}),on=['asset','start_time'],validate='one_to_one')
    assert len(joined)==len(scores)
    parts.append(joined)
data=pd.concat(parts,ignore_index=True)
train_mask=(data.start_time<pd.Timestamp('2026-08-01',tz='UTC')) if a.phase=='diagnostic' else np.ones(len(data),dtype=bool)
normalizers=data.loc[train_mask].groupby('asset').base.mean().to_dict()
data['gradient']=(data.plus-data.minus)/(.2*data.asset.map(normalizers))
data['target']=np.clip(-2*data.gradient,-3,3) if a.target=='gradient' else data.proxy_target
train=data.loc[train_mask];test=data[data.start_time>=pd.Timestamp('2026-06-01' if historical else '2026-08-01',tz='UTC')]
X=train[features].to_numpy();y=train.target.to_numpy();T=test[features].to_numpy()
mu=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1);Z=(X-mu)/sd
ridge=np.linalg.solve(Z.T@Z+np.eye(len(features))*len(X)*.03,Z.T@(y-y.mean()))
model=GradientBoostingRegressor(n_estimators=80,learning_rate=.05,max_depth=2,min_samples_leaf=100,loss='squared_error',random_state=seed)
model.fit(X,y);center=float(model.predict(X).mean())
trees=[]
for estimator in model.estimators_[:,0]:
    t=estimator.tree_
    trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,0,0].tolist()})
export={'features':features,'shape_table':'seasonality_apr01.json' if historical else 'seasonality_jul01.json','train_end_exclusive':{'diagnostic':'2026-08-01','refit':'2026-09-01','historical':'2026-07-01'}[a.phase],
        'learning_rate':.05,'offset':float(model.init_.constant_[0,0])-center,'trees':trees,
        'ridge':{'mean':mu.tolist(),'std':sd.tolist(),'coef':ridge.tolist()},
        'target':'minus_twice_official_log_width_CRPS_slope_normalized_by_training_asset_mean' if a.target=='gradient' else 'log_mean_absolute_return_proxy',
        'gradient_step':.1 if a.target=='gradient' else None,'target_clip':3 if a.target=='gradient' else 1.5,'normalizers':normalizers if a.target=='gradient' else {},'seed':seed}
def predict_export(values):
    out=np.full(len(values),export['offset'])
    for tree in trees:
        for i,x in enumerate(values.astype(np.float32)):
            node=0
            while tree['left'][node]!=-1:
                node=tree['left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
            out[i]+=.05*tree['value'][node]
    return out
error=float(np.max(np.abs(predict_export(T)-(model.predict(T)-center))))
assert error<1e-12,error
report={'phase':a.phase,'target':a.target,'train_origins':len(train),'heldout_origins':len(test) if a.phase=='diagnostic' else 0,'parity_rows':len(test),'train_start':str(train.start_time.min()),'train_last_origin':str(train.start_time.max()),
        'parity_start':str(test.start_time.min()),'training_asset_mean_CRPS':normalizers,'json_export_max_error':error,
        'decision':'pending actual official CRPS; gradient fit is not a scoring result',
        'evaluation_scope':'July fit for August holdout' if a.phase=='diagnostic' else 'refit without a new holdout'}
Path(f'agent/{stem}_{a.phase}.json').write_text(json.dumps(export,indent=2))
Path(f'agent/results/{stem}_{a.phase}_fit_report.json').write_text(json.dumps(report,indent=2))
data.to_parquet(f'agent/results/{stem}_{a.phase}_training_data.parquet',index=False)
print(json.dumps(report),flush=True)
