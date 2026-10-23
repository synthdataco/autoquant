"""Training-only constant control matched to adaptive log-volatility variance."""
import os,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from predict import load_minute_prices,context_end
from synth_lib.preparation.config import default_store_root
from research_model import simulate_config
p=argparse.ArgumentParser();p.add_argument('--start',default='2026-06-08');p.add_argument('--end',default='2026-09-01');p.add_argument('--table',default='seasonality.json');p.add_argument('--clock',default='clock.json');p.add_argument('--output',default='agent/matched_mix.json');p.add_argument('--training-output',default='agent/results/matched_mix_training.csv');p.add_argument('--no-queue',action='store_true');a=p.parse_args()
seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011'))
rows=[]
starts=pd.date_range(a.start,a.end,freq='6h',inclusive='left',tz='UTC')+pd.Timedelta(minutes=17)
for asset in ['BTC','ETH','SOL','XRP','HYPE']:
 frame=load_minute_prices(default_store_root(asset).parents[2],asset,starts[0]-pd.Timedelta(days=8),starts[-1])
 for t in starts:
  end=context_end(t,3600);context=frame.loc[(frame.index>end-pd.Timedelta(days=7))&(frame.index<=end)]
  state={};cfg={'seasonal':True,'clock':True,'fast_half':15,'table':a.table,'clock_table':a.clock,'_seed':seed,'_variance_diagnostics':state}
  simulate_config(asset,t,60,3600,1,context,np.random.default_rng(seed),cfg)
  ratio=state['initial']/state['long'];width=min(.55,float(np.sqrt(.35**2+.10*np.log(ratio)**2)))
  rows.append(dict(asset=asset,start_time=str(t),ratio=ratio,mix=width))
 df=pd.DataFrame([r for r in rows if r['asset']==asset]);print(asset,df['mix'].describe().to_dict(),flush=True)
df=pd.DataFrame(rows);matched=float(np.sqrt(df['mix'].pow(2).mean()))
Path(a.output).write_text(json.dumps({'train_start':a.start,'train_end_exclusive':a.end,'contexts':len(df),'constant_mix':matched},indent=2))
df.to_csv(a.training_output,index=False)
print('Matched constant mixture',matched,flush=True)
if a.no_queue:raise SystemExit(0)
job={'cmd':['uv','run','agent/run_suite.py','--tag','matched_mix_v7_full','--model','candidate_v8_research','--config',json.dumps({'mix':matched}),'--cached-prompts','--prediction-dtype','float32','--days','10','--ends','2026-09-19','2026-10-11','--assets','BTC','ETH','SOL','XRP','HYPE','--formats','1h','--workers','2'],'log':'agent/results/matched_mix_v7_full_suite.log'}
Path('agent/queue_matched_mix_score.json').write_text(json.dumps([job],indent=2))
