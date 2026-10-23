"""Aggregate retained official component totals; no score formulas."""
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('baseline');a=p.parse_args();rows=[]
for f in (Path('agent/results')/a.candidate).glob('*.parquet'):
 b=Path('agent/results')/a.baseline/f.name
 if not b.exists():continue
 asset,fmt,end=f.stem.split('_');x=pd.read_parquet(f);y=pd.read_parquet(b)
 keys=['asset','time_length','start_time','scored_time'];d=x.merge(y,on=keys,suffixes=('_new','_base'),validate='one_to_one')
 for suffix in ['new','base']:
  component=d['component_crps_'+suffix].map(json.loads)
  d['vol_'+suffix]=component.map(lambda c:c.get('Vol',0))
  d['price_'+suffix]=d['crps_'+suffix]-d['vol_'+suffix]
 for kind in ['crps','price','vol']:
  new=d[kind+'_new'].mean();old=d[kind+'_base'].mean()
  rows.append({'asset':asset,'end':end,'component':kind,'candidate':new,'baseline':old,'change':new-old,'pct':100*(new/old-1)})
d=pd.DataFrame(rows);print(d.round(5).to_string(index=False));print('\nEqual-asset component changes:\n',d.groupby(['end','component'])[['change','pct']].mean().round(5).to_string())
d.to_csv(Path('agent/results')/(a.candidate+'_components_vs_'+a.baseline+'.csv'),index=False)
