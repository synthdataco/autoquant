"""Train a small residual volatility correction beyond seasonal EWMA.

Fit only pre-validation prices. The target is future mean absolute return,
not a replacement implementation of CRPS. Promotion requires official backtests.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from research_model import shape_at,residual_features
from scipy.signal import lfilter

def finite_ewma(x,half,n=10078):
 values=x.to_numpy(dtype=float);ok=np.isfinite(values);lam=2**(-1/half)
 num=lfilter([1.],[1.,-lam],np.where(ok,values,0.));den=lfilter([1.],[1.,-lam],ok.astype(float))
 if len(x)>n:
  num[n:]-=lam**n*num[:-n].copy();den[n:]-=lam**n*den[:-n].copy()
 return pd.Series(num/np.maximum(den,1e-12),index=x.index)

p=argparse.ArgumentParser();p.add_argument('--start',default='2026-01-01');p.add_argument('--end',default='2026-09-01');p.add_argument('--output',default='agent/residual.json');a=p.parse_args()
assets=['BTC','ETH','SOL','XRP','HYPE'];rows=[]
for asset in assets:
 fs=[f for f in sorted(default_store_root(asset).glob('date=*.parquet')) if a.start<=f.stem[5:]<a.end]
 df=pd.concat([pd.read_parquet(f,columns=['timestamp','close','high','low','volume','trade_count']) for f in fs]).set_index('timestamp').sort_index()
 season=shape_at(df.index,asset,True);ret=np.log(df.close.astype(float)).diff();var=ret.pow(2)/season
 v={h:finite_ewma(var,h) for h in [15,60,360,1440,4320]}
 level=.6*v[60]+.25*v[360]+.15*v[4320]
 # Same mean reversion as simulator, averaged over the next hour.
 decay=np.exp2(-np.arange(1,61)/240).mean();forecast=v[4320]+(level-v[4320])*decay
 rangevar=finite_ewma(np.log(df.high.astype(float)/df.low.astype(float))**2/(4*np.log(2))/season,60,n=10079)
 volume=df.volume.astype(float);trades=df.trade_count.astype(float)
 eps=1e-16
 f=pd.DataFrame(index=df.index)
 f['fast_ratio']=np.log((v[15]+eps)/(v[60]+eps)).clip(-3,3)
 f['slow_ratio']=np.log((v[60]+eps)/(v[4320]+eps)).clip(-3,3)
 f['range_ratio']=np.log((rangevar+eps)/(v[60]+eps)).clip(-3,3)
 f['volume_ratio']=np.log((finite_ewma(volume,60,n=10079)+1)/(finite_ewma(volume,1440,n=10079)+1)).clip(-3,3)
 f['trade_ratio']=np.log((finite_ewma(trades,60,n=10079)+1)/(finite_ewma(trades,1440,n=10079)+1)).clip(-3,3)
 f['return_signed']=(ret.rolling(60,min_periods=50).sum()/np.sqrt(60*v[60]*season+eps)).clip(-4,4)
 # Only future bars, never the current return; all labels end before train_end.
 target=(ret.abs()/np.sqrt(season)).iloc[::-1].rolling(60,min_periods=50).mean().iloc[::-1].shift(-3)
 f['target']=np.log((target+1e-10)/(np.sqrt(2/np.pi)*np.sqrt(forecast+eps)))
 f=f.iloc[10080::30].dropna();f['asset']=asset
 sample=f.index[-100];position=df.index.get_loc(sample)
 serving=residual_features(df.iloc[position-10078:position+1],asset)
 error=max(abs(float(f.loc[sample,k])-v) for k,v in serving.items())
 if error>1e-7:raise AssertionError(f'{asset} training/serving feature mismatch {error}')
 rows.append(f)
 print(asset,len(f),'feature parity max error',error,flush=True)
data=pd.concat(rows);features=['fast_ratio','slow_ratio','range_ratio','volume_ratio','trade_ratio','return_signed']
train=data[data.index<'2026-08-01'];test=data[data.index>='2026-08-01']
models={};reports=[]
for name,cols in [('memory',['fast_ratio','slow_ratio']),('range',['range_ratio']),('volume',['volume_ratio','trade_ratio']),('leverage',['return_signed']),('all',features)]:
 def fit(d):
  X=d[cols].to_numpy();y=d.target.to_numpy();mu=X.mean(0);sd=np.maximum(X.std(0),.1);Z=np.column_stack([np.ones(len(X)),(X-mu)/sd]);reg=np.eye(Z.shape[1])*len(Z)*.03;reg[0,0]=0
  coef=np.linalg.solve(Z.T@Z+reg,Z.T@y)
  return mu,sd,coef
 mu,sd,beta=fit(train);X=np.column_stack([np.ones(len(test)),(test[cols].to_numpy()-mu)/sd]);pred=X@beta;y=test.target.to_numpy()
 baseline=float(np.mean((y-train.target.mean())**2));mse=float(np.mean((y-pred)**2))
 # Bound from width errors: one quarter of the variance explained in log scale.
 bound=.25*max(0,baseline-mse)
 reports.append(dict(name=name,train_n=len(train),test_n=len(test),proxy_mse_base=baseline,proxy_mse=mse,approx_crps_gain_bound=bound,coef=beta.tolist()))
 # Preserve diagnostic split model as well as refit on the entire pre-validation history.
 mu,sd,beta=fit(data);models[name]={'features':cols,'mean':mu.tolist(),'std':sd.tolist(),'coef':beta.tolist()}
 print(name,'proxy MSE',round(mse,6),'vs',round(baseline,6),'rough CRPS gain bound',round(bound*100,3),'%',flush=True)
Path(a.output).write_text(json.dumps({'train_start':a.start,'train_end_exclusive':a.end,'models':models},indent=2))
Path('agent/results/residual_diagnostics.json').write_text(json.dumps(reports,indent=2))
