"""Display paired daily official CRPS differences; no metric reimplementation."""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('baseline');a=p.parse_args();rows=[]
for f in (Path('agent/results')/a.candidate).glob('*.parquet'):
 b=Path('agent/results')/a.baseline/f.name
 if not b.exists():continue
 x=pd.read_parquet(f);y=pd.read_parquet(b)
 keys=['asset','time_length','start_time','scored_time']
 d=x[keys+['crps']].merge(y[keys+['crps']],on=keys,suffixes=('_new','_base'))
 d=d[(d.crps_new>0)&(d.crps_base>0)]
 d['day']=pd.to_datetime(d.scored_time).dt.strftime('%Y-%m-%d');rows.append(d)
if not rows:raise SystemExit('No paired results yet.')
x=pd.concat(rows).drop_duplicates(['asset','time_length','start_time','scored_time'])
x['config']=x.apply(lambda r:'CRYPTO_1H' if r.time_length==3600 else 'CRYPTO_24H' if r.asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'COM_EQU_24H',axis=1)
g=x.groupby(['config','asset','day'])[['crps_new','crps_base']].mean();g['change']=g.crps_new/g.crps_base-1
print(g.groupby(['config','day']).change.agg(['mean','min','max','count']).mul({'mean':100,'min':100,'max':100,'count':1}).round(4).to_string())
