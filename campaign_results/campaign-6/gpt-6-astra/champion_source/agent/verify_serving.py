"""Serving/reference parity and contract checks on actual sparse snapshot contexts."""
import argparse,importlib,json,os,time
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from predict import load_minute_prices,context_end
import modeling
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);a=p.parse_args();reference=importlib.import_module(a.reference)
crypto=['BTC','ETH','SOL','XRP','HYPE'];assets=crypto+['XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']
os.environ['MODEL_CONFIG']='{"scale":9}'
count=0;contexts={}
for date in ['2026-09-16','2026-10-08']:
 t=pd.Timestamp(date,tz='UTC')
 for asset in assets:
  f=load_minute_prices(default_store_root(asset).parents[2],asset,t-pd.Timedelta(days=8),t)
  for length,step in ([(86400,300),(3600,60)] if asset in crypto else [(86400,300)]):
   end=context_end(t,length);frame=f.loc[t-pd.Timedelta(days=7):end]
   x=modeling.simulate(asset,t,step,length,17,frame);y=reference.simulate(asset,t,step,length,17,frame)
   assert x==y,(asset,date,length)
   assert x[0]==t and x[1]==step
   paths=np.asarray(x[2:]);assert paths.shape==(17,length//step+1)
   assert np.isfinite(paths).all() and (paths>0).all()
   assert np.all(paths[:,0]==frame.close.dropna().iloc[-1]);count+=1
   contexts[(asset,length)]=(t,step,frame)
latencies={}
for asset,length in [('BTC',3600),('AAPLX',86400),('XAU',86400)]:
 t,step,frame=contexts[(asset,length)];start=time.perf_counter()
 x=modeling.simulate(asset,t,step,length,1000,frame);elapsed=time.perf_counter()-start
 assert len(x)==1002 and np.asarray(x[2:]).shape==(1000,length//step+1)
 assert np.isfinite(np.asarray(x[2:])).all() and (np.asarray(x[2:])>0).all()
 assert x==reference.simulate(asset,t,step,length,1000,frame)
 y=modeling.simulate(asset,t,step,length,1000,frame);assert x==y
 latencies[asset+'_'+str(length)]={'seconds':elapsed,'context_nan_closes':int(frame.close.isna().sum())}
t,step,frame=contexts[('BTC',3600)]
os.environ['SYNTH_BENCHMARK_SEED']='11';x=modeling.simulate('BTC',t,step,3600,17,frame)
os.environ['SYNTH_BENCHMARK_SEED']='37';y=modeling.simulate('BTC',t,step,3600,17,frame)
assert x!=y
print(json.dumps({'reference':a.reference,'exact_parity_calls':count,'arbitrary_count':17,'contract_count':1000,'repeatability':True,'seed_changes_paths':True,'latencies':latencies},indent=2))
