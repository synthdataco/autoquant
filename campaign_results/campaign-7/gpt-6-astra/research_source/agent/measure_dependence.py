"""Descriptive horizon scaling and return dependence, without computing CRPS."""
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd
from explore import load_asset
from run_suite import ROOT,CRYPTO,OTHER


def measure(asset):
    frame=load_asset(asset)
    p=np.log(frame.close)
    r=p.diff()
    output=[]
    for horizon in [1,5,30,60,180,1440]:
        changes=p.diff(horizon)
        rv=(r*r).rolling(horizon,min_periods=horizon).sum()
        ratio=changes/np.sqrt(rv.where(rv>0))
        data=pd.DataFrame({'r2':changes**2,'rv':rv,'abs_ratio':ratio.abs(),'ratio':ratio})
        data=data.where(np.isfinite(data)).dropna()
        for month,group in data.groupby(data.index.strftime('%Y-%m')):
            output.append({'asset':asset,'month':month,'horizon':horizon,'n':len(group),
                           'variance_ratio':group.r2.sum()/group.rv.sum(),
                           'absolute_efficiency':group.abs_ratio.mean()*np.sqrt(np.pi/2),
                           'q95_abs_ratio':group.abs_ratio.quantile(.95)})
    return pd.DataFrame(output)


if __name__=='__main__':
    os.chdir(ROOT)
    with ProcessPoolExecutor(max_workers=2) as pool:
        data=pd.concat(list(pool.map(measure,CRYPTO+OTHER)),ignore_index=True)
    data.to_csv(ROOT/'agent/results/descriptive/dependence.csv',index=False)
    view=data[data.month>='2026-06'].groupby(['asset','horizon'])[['variance_ratio','absolute_efficiency']].median()
    print(view.round(4).to_string())
