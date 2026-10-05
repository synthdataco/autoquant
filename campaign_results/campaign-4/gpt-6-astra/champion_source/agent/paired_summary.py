"""Paired daily differences; day-cluster uncertainty acknowledges overlapping prompts."""
import sys
import numpy as np
import pandas as pd
from synth.validator.moving_average import ASSET_COEFFICIENTS
crypto={'BTC','ETH','SOL','XRP','HYPE'}
a=pd.read_csv('agent/results/'+sys.argv[1]+'.csv');b=pd.read_csv('agent/results/'+sys.argv[2]+'.csv')
d=a.merge(b,on=['asset','horizon','start'],suffixes=('_a','_b'),validate='one_to_one')
assert np.allclose(d.coverage_a,d.coverage_b)
d['config']=['CRYPTO_1H' if h==3600 else 'CRYPTO_24H' if x in crypto else 'COM_EQU_24H' for x,h in zip(d.asset,d.horizon)]
d['day']=d.start.str[:10];d['coef']=d.asset.map(ASSET_COEFFICIENTS)
d['diff']=d.crps_b-d.crps_a
rng=np.random.default_rng(123)
for config,g in d.groupby('config'):
 daily=g.groupby('day').apply(lambda x:np.average(x['diff'],weights=x.coef),include_groups=False).to_numpy()
 reps=np.mean(rng.choice(daily,size=(10000,len(daily))),axis=1)
 base=np.average(g.crps_a,weights=g.coef);new=np.average(g.crps_b,weights=g.coef)
 print(config,'n',len(g),'weighted CRPS',round(base,4),round(new,4),'change%',round((new/base-1)*100,3),'daily bootstrap difference 95%',np.quantile(reps,[.025,.975]).round(4).tolist())
