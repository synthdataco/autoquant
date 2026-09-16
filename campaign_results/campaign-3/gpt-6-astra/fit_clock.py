"""Fit strictly positive intraday variance-shape priors from pre-field prices."""

import json

import numpy as np
import pandas as pd

from predict import AGENT_DIR, load_context, store_root
from research import CRYPTO,EQUITY

profiles={}
for asset in CRYPTO+EQUITY:
    first=sorted(store_root(asset).glob('date=*.parquet'))[0].stem.removeprefix('date=')
    start=max(pd.Timestamp(first,tz='UTC'),pd.Timestamp('2026-04-01',tz='UTC'))
    close=load_context(asset,start,pd.Timestamp('2026-07-26',tz='UTC')).close.dropna()
    gaps=np.diff(close.index.as_unit('ns').asi8)/60e9
    absolute=np.abs(np.diff(np.log(close.to_numpy(dtype=float))))/np.sqrt(gaps)
    frame=pd.DataFrame({'absolute':absolute,'day':close.index[1:].date,'hour':close.index[1:].hour})
    frame['relative']=frame.absolute/frame.groupby('day').absolute.transform('mean').clip(lower=1e-9)
    profile=frame.groupby('hour').relative.mean().reindex(range(24),fill_value=1).to_numpy()
    profile=.25*np.roll(profile,1)+.5*profile+.25*np.roll(profile,-1)
    profile=np.clip(profile,.25,3)
    profiles[asset]=profile/profile.mean()
    print(asset,profile.min(),profile.max(),flush=True)
np.savez(AGENT_DIR/'clock_profiles.npz',**profiles)
(AGENT_DIR/'clock_profiles.json').write_text(json.dumps({'fitting_start':'2026-04-01 or first available trade','fitting_end':'2026-07-26','interpretation':'Positive hourly volatility shape only; no market-closure/calendar rules.'},indent=2))
