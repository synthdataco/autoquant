from pathlib import Path
import pandas as pd
rows=[]
for p in Path('agent/results').glob('cal_*.csv'):
 if not p.with_suffix('.json').exists():continue
 d=pd.read_csv(p);d['config']=['CRYPTO_1H' if h==3600 else 'CRYPTO_24H' if a in ['BTC','ETH','SOL','XRP','HYPE'] else 'COM_EQU_24H' for a,h in zip(d.asset,d.horizon)]
 scale=float(p.stem.split('_s')[1].split('_')[0].replace('p','.'));mix=float(p.stem.split('_m')[1].replace('p','.'))
 for c,g in d.groupby('config'):rows.append(dict(configuration=c,scale=scale,mixture=mix,crps=g.groupby('asset').crps.mean().mean()))
f=pd.DataFrame(rows)
if len(f):
 f.to_csv('agent/results/calibration_summary.csv',index=False)
 for c,g in f.groupby('configuration'):print(c,'\n',g.pivot(index='scale',columns='mixture',values='crps').round(4).to_string())
