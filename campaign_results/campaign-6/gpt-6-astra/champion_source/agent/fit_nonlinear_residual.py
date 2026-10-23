"""Chronological proxy screen; never computes or substitutes for official CRPS."""
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from synth_lib.preparation.config import default_store_root
from predict import context_end
from research_model import residual_features,simulate_config

p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['diagnostic','refit','historical'],default='diagnostic')
a=p.parse_args()
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
shape='seasonality.json' if a.phase=='refit' else 'seasonality_jul01.json'
clock='clock.json' if a.phase=='refit' else 'clock_jul01.json'
events='event_multipliers.json' if a.phase=='refit' else 'event_multipliers_jul01.json'
cutoff='2026-07-01' if a.phase=='historical' else '2026-09-01'
features=['fast_ratio','slow_ratio','range_ratio','volume_ratio','trade_ratio','return_signed']
dataset=Path(f'agent/results/nonlinear_{a.phase}_data.parquet')
if dataset.exists():
    data=pd.read_parquet(dataset)
else:
    rows=[]
    for asset in ['BTC','ETH','SOL','XRP','HYPE']:
        files=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if '2026-03-24'<=f.stem[5:]<cutoff]
        frame=pd.concat([pd.read_parquet(f,columns=['timestamp','close','high','low','volume','trade_count']) for f in files]).set_index('timestamp').sort_index()
        returns=np.log(frame.close.astype(float)).diff()
        origins=pd.date_range('2026-04-01',pd.Timestamp(cutoff)-pd.Timedelta(hours=3),freq='3h',tz='UTC')
        for i,origin in enumerate(origins):
            t=origin+pd.Timedelta(minutes=(17*i+2)%60)
            if t+pd.Timedelta(hours=1)>=pd.Timestamp(cutoff,tz='UTC'):continue
            context=frame.loc[t-pd.Timedelta(days=7):context_end(t,3600)]
            future=pd.date_range(t+pd.Timedelta(minutes=1),periods=60,freq='min')
            actual=returns.reindex(future).abs()
            if actual.notna().sum()<55:continue
            sink={}
            config={'seasonal':True,'clock':True,'events':True,'event_power':1.5,'event_bar_alignment':True,
                    'mix':.35,'fast_half':15,'seasonal_shrink':.1,'scale':1.,'sampler':'sobol_bridge',
                    'table':shape,'clock_table':clock,'event_table':events,
                    '_calendar_path':os.environ.get('SYNTH_BENCHMARK_CALENDAR',''),'_seed':seed,'_residual_training':sink}
            simulate_config(asset,t,60,3600,17,context,np.random.default_rng(seed),config)
            predicted=np.sqrt(2/np.pi)*np.exp(-.5*.35**2)*np.mean(sink['sig'][actual.notna().to_numpy()])
            row=residual_features(context,asset,shape)
            row.update(asset=asset,start_time=t,target=float(np.clip(np.log((actual.mean()+1e-12)/(predicted+1e-12)),-1.5,1.5)))
            rows.append(row)
        print(asset,'rows',len(rows),flush=True)
    data=pd.DataFrame(rows);data.to_parquet(dataset,index=False)

train=data[data.start_time<'2026-08-01'] if a.phase=='diagnostic' else data
test=data[data.start_time>=('2026-06-01' if a.phase=='historical' else '2026-08-01')]
X=train[features].to_numpy();y=train.target.to_numpy()
mu=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.1)
Z=(X-mu)/sd
ridge=np.linalg.solve(Z.T@Z+np.eye(len(features))*len(X)*.03,Z.T@(y-y.mean()))
model=GradientBoostingRegressor(n_estimators=80,learning_rate=.05,max_depth=2,min_samples_leaf=100,loss='squared_error',random_state=seed)
model.fit(X,y)
center=float(model.predict(X).mean())
T=test[features].to_numpy();target=test.target.to_numpy()-y.mean()
tree_pred=model.predict(T)-center;ridge_pred=(T-mu)/sd@ridge
report={'phase':a.phase,'train_rows':len(train),'test_rows':len(test),'zero_mse':float(np.mean(target**2)),
        'ridge_mse':float(np.mean((target-ridge_pred)**2)),'tree_mse':float(np.mean((target-tree_pred)**2))}
report['evaluation_scope']='chronological_holdout' if a.phase=='diagnostic' else 'training_subset_not_validation'
report['proxy_gate_pass']=(report['tree_mse']<min(report['zero_mse'],report['ridge_mse'])) if a.phase=='diagnostic' else None
trees=[]
for estimator in model.estimators_[:,0]:
    tree=estimator.tree_
    trees.append({'left':tree.children_left.tolist(),'right':tree.children_right.tolist(),'feature':tree.feature.tolist(),'threshold':tree.threshold.tolist(),'value':tree.value[:,0,0].tolist()})
export={'features':features,'shape_table':shape,'train_end_exclusive':'2026-08-01' if a.phase=='diagnostic' else cutoff,
        'learning_rate':.05,'offset':float(model.init_.constant_[0,0])-center,'trees':trees,
        'ridge':{'mean':mu.tolist(),'std':sd.tolist(),'coef':ridge.tolist()}}
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
report['json_export_max_error']=error
Path(f'agent/nonlinear_{a.phase}.json').write_text(json.dumps(export,indent=2))
Path(f'agent/results/nonlinear_{a.phase}_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
