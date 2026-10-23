"""Aggregate matched official CRPS across independent ensemble seeds."""
import json,argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('--base',default='candidate_v3');a=p.parse_args();rows=[]
for seed in [20261011,11,37]:
 for f in Path(f'agent/results/{a.prefix}_seed{seed}').glob('*.json'):
  b=Path(f'agent/results/{a.base}_seed{seed}')/f.name
  if not b.exists():continue
  x=json.loads(f.read_text());y=json.loads(b.read_text());assert x['num_prompts']==y['num_prompts']
  asset,fmt,end=f.stem.split('_');rows.append({'seed':seed,'asset':asset,'fmt':fmt,'end':end,'pct':100*(x['mean_crps']/y['mean_crps']-1)})
d=pd.DataFrame(rows);print(d.groupby(['seed','end']).pct.agg(['mean',lambda x:int((x<0).sum()),'count']).to_string());print('Per seed:',d.groupby('seed').pct.mean().to_dict());print('Pooled:',d.pct.mean(),'wins',int((d.pct<0).sum()),'/',len(d))
d.to_csv('agent/results/'+a.prefix+'_seed_comparison.csv',index=False)
