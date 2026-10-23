"""Descriptive paired block intervals on already-official CRPS values."""
import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('baseline');a=p.parse_args();rows=[]
for f in (Path('agent/results')/a.candidate).glob('*.parquet'):
 b=Path('agent/results')/a.baseline/f.name
 if not b.exists():continue
 keys=['asset','time_length','start_time','scored_time']
 x=pd.read_parquet(f);y=pd.read_parquet(b)
 d=x[keys+['crps']].merge(y[keys+['crps']],on=keys,suffixes=('_new','_base'),validate='one_to_one')
 d['day']=pd.to_datetime(d.scored_time).dt.strftime('%Y-%m-%d');d['window']=f.stem.rsplit('_',1)[1];rows.append(d)
x=pd.concat(rows)
x['config']=np.where(x.time_length==3600,'CRYPTO_1H',np.where(x.asset.isin(['BTC','ETH','SOL','XRP','HYPE']),'CRYPTO_24H','COM_EQU_24H'))
g=x.groupby(['config','window','day','asset'])[['crps_new','crps_base']].mean();g['pct']=(g.crps_new/g.crps_base-1)*100
d=g.groupby(['config','window','day']).pct.mean().reset_index()
rng=np.random.default_rng(int(os.environ.get('SYNTH_BENCHMARK_SEED','20261011')));results=[]
for (config,window),v in d.groupby(['config','window']):
 y=v.sort_values('day').pct.to_numpy();n=len(y)
 starts=rng.integers(0,n,size=(10000,(n+1)//2));indices=np.stack([starts,(starts+1)%n],axis=-1).reshape(10000,-1)[:,:n]
 boot=y[indices].mean(axis=1)
 results.append({'config':config,'window':window,'days':n,'daily_mean_pct':float(y.mean()),'days_improved':int((y<0).sum()),'block2_interval95':np.quantile(boot,[.025,.975]).tolist()})
print(json.dumps(results,indent=2))
(Path('agent/results')/(a.candidate+'_block_intervals.json')).write_text(json.dumps(results,indent=2))
