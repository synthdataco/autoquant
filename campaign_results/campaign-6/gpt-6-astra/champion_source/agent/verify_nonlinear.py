"""Verify current-base parity and JSON residual serving on real contexts."""
import os,json,time,argparse
from pathlib import Path
import numpy as np,pandas as pd
from synth_lib.preparation.config import default_store_root
from predict import load_minute_prices,context_end
from candidate_v9 import simulate as reference
from research_model import simulate_config,residual_features
from nonlinear_residual import correction,parameters
p=argparse.ArgumentParser();p.add_argument('--artifact',default='nonlinear_refit.json');p.add_argument('--output',default='agent/results/nonlinear_serving_gate.json');a=p.parse_args()
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
rows=[]
for asset in ['BTC','ETH','HYPE']:
    t=pd.Timestamp('2026-09-16T12:17:00Z')
    frame=load_minute_prices(default_store_root(asset).parents[2],asset,t-pd.Timedelta(days=8),t)
    frame=frame.loc[t-pd.Timedelta(days=7):context_end(t,3600)]
    sink={}
    c={'seasonal':True,'clock':True,'events':True,'event_power':1.5,'event_bar_alignment':True,
       'mix':.35,'fast_half':15,'seasonal_shrink':.1,'scale':1.,'sampler':'sobol_bridge',
       'table':'seasonality.json','clock_table':'clock.json','event_table':'event_multipliers.json',
       '_calendar_path':os.environ.get('SYNTH_BENCHMARK_CALENDAR',''),'_seed':seed,'_residual_training':sink}
    actual=simulate_config(asset,t,60,3600,17,frame,np.random.default_rng(seed),c)
    assert actual==reference(asset,t,60,3600,17,frame)
    for kind in ['tree','ridge']:
        c['nonlinear']=kind
        c['nonlinear_model']=a.artifact
        began=time.perf_counter()
        result=simulate_config(asset,t,60,3600,1000,frame,np.random.default_rng(seed),c)
        elapsed=time.perf_counter()-began
        assert result==simulate_config(asset,t,60,3600,1000,frame,np.random.default_rng(seed),c)
        paths=np.asarray(result[2:]);assert paths.shape==(1000,61) and np.all(np.isfinite(paths)) and np.all(paths>0)
        assert np.all(paths[:,0]==frame.close.dropna().iloc[-1])
        value=correction(frame,asset,name=a.artifact,kind=kind)
        rows.append({'asset':asset,'kind':kind,'base_exact':True,'repeatable':True,'paths':1000,'seconds':elapsed,'raw_correction':value})
Path(a.output).write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
