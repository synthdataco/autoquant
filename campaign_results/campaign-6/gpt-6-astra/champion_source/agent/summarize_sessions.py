"""Paired official CRPS split by forecast-origin New York weekday/weekend.

A 24h path can span both regimes; this is descriptive origin stratification,
not an attribution of each interval's score to the session it covers.
"""
import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--base',required=True);a=p.parse_args()
rows=[]
for f in (Path('agent/results')/a.candidate).glob('*.parquet'):
 b=Path('agent/results')/a.base/f.name
 if not b.exists():continue
 x=pd.read_parquet(f);y=pd.read_parquet(b)
 keys=['asset','time_length','start_time','scored_time']
 d=x[keys+['crps']].merge(y[keys+['crps']],on=keys,suffixes=('_new','_base'),validate='one_to_one')
 if len(d)!=len(x) or len(d)!=len(y):raise ValueError('Unequal prompt sets: '+f.name)
 d['weekend_origin']=pd.to_datetime(d.start_time,utc=True).dt.tz_convert('America/New_York').dt.dayofweek>=5
 d['window']=f.stem.rsplit('_',1)[1];rows.append(d)
if not rows:raise SystemExit('No paired artifacts')
d=pd.concat(rows).groupby(['window','asset','weekend_origin'])[['crps_new','crps_base']].mean().reset_index()
d['change_pct']=100*(d.crps_new/d.crps_base-1)
print(d.groupby(['window','weekend_origin']).agg(mean_pct=('change_pct','mean'),wins=('change_pct',lambda x:int((x<0).sum())),cells=('change_pct','size')).to_string())
d.to_csv(Path('agent/results')/(a.candidate+'_sessions.csv'),index=False)
