"""Plot official evaluation summaries; no metric reimplementation."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--candidate',default='candidate_v1_full');p.add_argument('--baseline',default='baseline_full');p.add_argument('--candidate-com');p.add_argument('--candidate-short');p.add_argument('--output-name');a=p.parse_args();rows=[]
for f in (Path('agent/results')/a.candidate).glob('*.json'):
 if a.candidate_short and '_1h_' in f.name:
  f=Path('agent/results')/a.candidate_short/f.name
 if a.candidate_com and f.stem.split('_')[0] not in ['BTC','ETH','SOL','XRP','HYPE']:
  f=Path('agent/results')/a.candidate_com/f.name
 b=Path('agent/results')/a.baseline/f.name
 if not b.exists():continue
 asset,fmt,end=f.stem.split('_');x=json.loads(f.read_text());y=json.loads(b.read_text())
 group='Crypto 1h' if fmt=='1h' else 'Crypto 24h' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'Commodities/equities 24h'
 rows.append({'end':end,'group':group,'reduction':100*(1-x['mean_crps']/y['mean_crps'])})
d=pd.DataFrame(rows).groupby(['group','end']).reduction.mean().unstack().reindex(['Crypto 24h','Commodities/equities 24h','Crypto 1h'])
fig,ax=plt.subplots(figsize=(10,5.4));colors=['#3366a5','#23866a'];positions=np.arange(3)
for j,end in enumerate(d.columns):
 bars=ax.bar(positions+(j-.5)*.34,d[end],width=.32,color=colors[j],label={'2026-09-19':'Sep 09–18','2026-10-11':'Oct 01–10'}.get(end,end))
 ax.bar_label(bars,fmt='%.2f%%',padding=4,fontsize=10)
ax.set_xticks(positions,d.index);ax.set_ylabel('CRPS reduction (%)');ax.set_ylim(0,float(d.max().max())*1.2);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False)
ax.set_title((a.output_name or a.candidate)+' versus seeded starter',loc='left',fontweight='bold',pad=18)
fig.text(.08,.025,'1000 paths · Equal-asset mean percentage changes · Development-inclusive validation windows',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.06,1,1]);out=Path('agent/figures');out.mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(out/((a.output_name or a.candidate)+'.'+ext),dpi=160)
print('Wrote',out/((a.output_name or a.candidate)+'.png'))
