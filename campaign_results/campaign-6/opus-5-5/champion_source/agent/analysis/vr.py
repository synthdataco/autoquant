import pandas as pd, numpy as np, sys
sys.path.insert(0,'/workspace/agent/analysis')
from season import load
for a in ['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']:
    c=load(a)['2026-07-01':]
    l=np.log(c)
    r1=l.diff(); r5=l.diff(5); r60=l.diff(60)
    v1=np.nanmean(r1**2); v5=np.nanmean(r5**2); v60=np.nanmean(r60**2)
    z1=(r1==0).sum()/r1.notna().sum()
    k=np.nanmean(r5**4)/v5**2
    print(f'{a:7s} VR5={v5/5/v1:.2f} VR60={v60/60/v1:.2f} zero1m={z1:.2%} kurt5={k:.1f} vol5bp={np.sqrt(v5)*1e4:.1f}')
