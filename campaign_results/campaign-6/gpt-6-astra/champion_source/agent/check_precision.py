"""Quantify storage-precision effects using already-official paired scores."""
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np
p=argparse.ArgumentParser();p.add_argument('float32_tag');p.add_argument('float64_tag');a=p.parse_args();rows=[]
for f in (Path('agent/results')/a.float32_tag).glob('*.parquet'):
 b=Path('agent/results')/a.float64_tag/f.name
 if not b.exists():continue
 x=pd.read_parquet(f);y=pd.read_parquet(b);keys=['asset','time_length','start_time','scored_time']
 d=x.merge(y,on=keys,suffixes=('_32','_64'),validate='one_to_one');assert len(d)==len(x)==len(y)
 delta=d.crps_32-d.crps_64
 rows.append({'cell':f.stem,'mean_delta':delta.mean(),'mean_pct':100*(d.crps_32.mean()/d.crps_64.mean()-1),'max_abs_prompt_delta':delta.abs().max(),'p99_abs_prompt_delta':delta.abs().quantile(.99)})
x=pd.DataFrame(rows);x.to_csv(Path('agent/results')/(a.float32_tag+'_precision.csv'),index=False)
print('cells',len(x));print('max absolute cell mean percent change',x.mean_pct.abs().max());print('max absolute prompt delta',x.max_abs_prompt_delta.max());print(x.round(9).to_string(index=False))
