"""Compare official CRPS summaries, preserving asset/window cells."""
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('tags',nargs='+');p.add_argument('--base',default='baseline');p.add_argument('--base-com');p.add_argument('--base-short');a=p.parse_args()
rows=[]
for tag in list(dict.fromkeys([a.base]+([a.base_com] if a.base_com else [])+([a.base_short] if a.base_short else [])+a.tags)):
 for f in (Path('agent/results')/tag).glob('*.json'):
  d=json.loads(f.read_text());asset,fmt,end=f.stem.split('_');rows.append(dict(tag=tag,asset=asset,fmt=fmt,end=end,crps=d['mean_crps'],n=d['num_prompts'],coverage=d['realized_coverage'],config='CRYPTO_1H' if fmt=='1h' else 'CRYPTO_24H' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'COM_EQU_24H'))
df=pd.DataFrame(rows)
if df.empty:raise SystemExit('no results')
controls=df[df.tag==a.base]
if a.base_com:controls=pd.concat([controls[controls.config!='COM_EQU_24H'],df[(df.tag==a.base_com)&(df.config=='COM_EQU_24H')]])
if a.base_short:controls=pd.concat([controls[controls.config!='CRYPTO_1H'],df[(df.tag==a.base_short)&(df.config=='CRYPTO_1H')]])
controls=controls.set_index(['asset','fmt','end']);base=controls.crps
if a.base_com or a.base_short:print('Controls:',a.base,'default;',a.base_com,'COM override;',a.base_short,'1h override')
for tag in a.tags:
 d=df[df.tag==tag].copy();d['base']=[base.get((r.asset,r.fmt,r.end),float('nan')) for r in d.itertuples()];d['change']=d.crps/d.base-1;d=d[d.base.notna()]
 for r in d.itertuples():
  if r.n!=controls.loc[(r.asset,r.fmt,r.end),'n']:raise ValueError('Mismatched prompt counts: '+str((tag,r.asset,r.fmt,r.end)))
 print('\n',tag,'cells',len(d))
 print(d.groupby(['end','config']).agg(crps=('crps','mean'),base=('base','mean'),pct=('change',lambda x:100*x.mean()),wins=('change',lambda x:(x<0).sum()),cells=('change','count')).round(4).to_string())
 print(d.pivot_table(index=['asset','fmt'],columns='end',values='change').mul(100).round(2).to_string())
