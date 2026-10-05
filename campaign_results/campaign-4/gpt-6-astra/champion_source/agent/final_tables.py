"""Reproducible report tables from persisted official scores."""
import json
from pathlib import Path
import pandas as pd,numpy as np
from synth.validator.moving_average import ASSET_COEFFICIENTS
crypto={'BTC','ETH','SOL','XRP','HYPE'}
def config(a,h):return 'CRYPTO_1H' if h==3600 else 'CRYPTO_24H' if a in crypto else 'COM_EQU_24H'
rows=[]
for model in ['baseline','candidate']:
 for p in Path('agent/results').glob(model+'_nine_full_*.json'):
  d=json.loads(p.read_text());asset,h=p.stem.split('_full_')[1].rsplit('_',1);h=int(h)
  rows.append(dict(model=model,asset=asset,horizon=h,configuration=config(asset,h),crps=d['mean_crps'],prompts=d['num_prompts'],coverage=d['realized_coverage'],coef=ASSET_COEFFICIENTS[asset]))
x=pd.DataFrame(rows)
if len(x):
 x.to_csv('agent/results/full_asset_table.csv',index=False)
 s=x.groupby(['configuration','model']).agg(crps=('crps','mean'),assets=('asset','nunique'),prompts=('prompts','sum'),coverage=('coverage','mean')).reset_index()
 s.to_csv('agent/results/full_configuration_table.csv',index=False);print(s.to_string(index=False))
 for c,g in x.groupby('configuration'):
  print(c,'coefficient-weighted',g.groupby('model').apply(lambda t:np.average(t.crps,weights=t.coef),include_groups=False).to_dict())
