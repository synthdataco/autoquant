import sys
import pandas as pd
from pathlib import Path
crypto={'BTC','ETH','SOL','XRP','HYPE'}
frames=[]
for name in sys.argv[1:]:
 if not (Path('agent/results')/(name+'.json')).exists():
  print('SKIPPING INCOMPLETE:',name);continue
 p=Path('agent/results')/(name+'.csv');d=pd.read_csv(p)
 d['configuration']=[('CRYPTO_1H' if h==3600 else 'CRYPTO_24H' if a in crypto else 'COM_EQU_24H') for a,h in zip(d.asset,d.horizon)]
 d['tag']=name;frames.append(d)
df=pd.concat(frames)
print(df.groupby(['asset','horizon','tag']).crps.mean().unstack().round(3).to_string())
print('\nConfiguration CRPS (equal asset weighting):')
print(df.groupby(['configuration','asset','tag']).crps.mean().groupby(['configuration','tag']).mean().unstack().round(4).to_string())
print('\nCoverage:',df.groupby('tag').coverage.mean().to_dict())
