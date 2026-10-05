import pandas as pd
from pathlib import Path
rows=[]
for window in ['aug1','aug2']:
 for seed in ['123','987']:
  for name in ['har48','har48_ma']:
   tag=f'{name}_{window}_s{seed}'
   d=pd.read_csv(Path('agent/results')/(tag+'.csv'))
   d['config']=['CRYPTO_1H' if h==3600 else 'CRYPTO_24H' if a in ['BTC','ETH','SOL','XRP','HYPE'] else 'COM_EQU_24H' for a,h in zip(d.asset,d.horizon)]
   for c,g in d.groupby('config'):rows.append(dict(window=window,seed=seed,model=name,configuration=c,crps=g.groupby('asset').crps.mean().mean()))
x=pd.DataFrame(rows);print(x.pivot(index=['configuration','window','seed'],columns='model',values='crps').round(5).to_string())
x.to_csv('agent/results/seed_summary.csv',index=False)
