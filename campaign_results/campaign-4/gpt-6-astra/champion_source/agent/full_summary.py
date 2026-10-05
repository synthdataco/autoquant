import json
from pathlib import Path
import pandas as pd
rows=[]
for p in Path('agent/results').glob('*_nine_full_*.json'):
 d=json.loads(p.read_text());parts=p.stem.split('_full_');asset,h=parts[1].rsplit('_',1)
 rows.append(dict(model=parts[0],asset=asset,horizon=int(h),crps=d['mean_crps'],n=d['num_prompts'],coverage=d.get('realized_coverage')))
x=pd.DataFrame(rows)
if len(x):
 print(x.pivot(index=['asset','horizon'],columns='model',values='crps').round(4).to_string())
 print('Coverage',x.groupby('model').coverage.min().to_dict())
 x.to_csv('agent/results/full_summary.csv',index=False)
