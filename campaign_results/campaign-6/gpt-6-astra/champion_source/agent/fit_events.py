"""Pooled event multipliers, with event dates as independent sampling units."""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--fine',action='store_true');ap.add_argument('--end',default='2026-09-01');ap.add_argument('--suffix',default=None);args=ap.parse_args();suffix=args.suffix if args.suffix is not None else ('_fine' if args.fine else '')
x=pd.read_csv(f'agent/results/event_diagnostics{suffix}.csv')
x=x[pd.to_datetime(x.time,utc=True)<pd.Timestamp(args.end,tz='UTC')]
x['group']=np.where(x.asset.isin(['BTC','ETH','SOL','XRP','HYPE']),'crypto','other')
x['ratio']=x.ratio.clip(.02,100)
out={}; diagnostics=[]
for (group,event,lo),g in x.groupby(['group','event','lo']):
 if event in ['EARNINGS','NYSE_CLOSED','NYSE_EARLY_CLOSE'] or lo==120:continue
 # Assets at one release share a shock: collapse to one datum per release.
 z=g.groupby('time').ratio.median()
 if len(z)<4:continue
 estimate=float(np.exp(.67*np.log(z).median()))
 estimate=max(1.,min(12.,estimate))
 loeo=[float(np.exp(.67*np.log(z.drop(i)).median())) for i in z.index]
 diagnostics.append(dict(group=group,event=event,lo=int(lo),n_dates=len(z),mult=estimate,leave_one_out_min=min(loeo),leave_one_out_max=max(loeo)))
 out.setdefault(group,{}).setdefault(event,{})[str(lo)]=estimate
Path(f'agent/event_multipliers{suffix}.json').write_text(json.dumps({'train_end_exclusive':args.end,'groups':out},indent=2))
pd.DataFrame(diagnostics).to_csv(f'agent/results/event_fit{suffix}.csv',index=False)
print(pd.DataFrame(diagnostics).round(3).to_string(index=False))
