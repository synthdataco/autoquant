import json
from pathlib import Path
import pandas as pd,numpy as np
new=Path('agent/results/score_seasonal/BTC_1h_2026-09-16.parquet');base=Path('agent/results/seasonal')/new.name
x=pd.read_parquet(new);y=pd.read_parquet(base);keys=['asset','time_length','start_time','scored_time']
d=x.merge(y,on=keys,suffixes=('_new','_base'),validate='one_to_one')
assert len(d)==len(x)==len(y)
assert np.array_equal(d.crps_new,d.crps_base)
assert all(json.loads(r.component_crps)['Overall']==r.crps for r in x.itertuples())
print('Official component recorder parity: exact on',len(d),'prompts')
