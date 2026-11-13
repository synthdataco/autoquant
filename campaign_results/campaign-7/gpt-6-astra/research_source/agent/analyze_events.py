"""Measure supplied-calendar event effects against matched earlier local times."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import numpy as np
import pandas as pd

from explore import load_asset
from run_suite import ROOT, CRYPTO, OTHER

WINDOWS=[(-30,-5),(-5,0),(0,1),(1,5),(5,15),(15,60),(60,180)]


def asset_effects(asset):
    frame=load_asset(asset)
    r=np.log(frame.close).diff()
    squared=r*r
    cal=pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
    cal=cal[(cal.scope=='all')|cal.scope.str.split(',').map(lambda x:asset in x)]
    cal=cal[(cal.time_utc<='2026-11-08 23:59:00+00:00')&(~cal.event.str.startswith('NYSE'))]
    rows=[]
    for time,events in cal.groupby('time_utc'):
        if time<frame.index.min()+pd.Timedelta(days=21) or time+pd.Timedelta(hours=3)>frame.index.max():
            continue
        types='|'.join(sorted(events.event.unique()))
        ny=pd.Timestamp(time).tz_convert('America/New_York')
        # Matched local weekday/time from prior weeks, excluding another event.
        controls=[]
        for weeks in range(1,9):
            local=ny.tz_localize(None)-pd.Timedelta(weeks=weeks)
            t=local.tz_localize('America/New_York').tz_convert('UTC')
            near=(pd.to_datetime(cal.time_utc)-t).abs()<pd.Timedelta(hours=3)
            if t>=frame.index.min()+pd.Timedelta(days=3) and not near.any():
                controls.append(t)
        if len(controls)<2:
            continue
        # Each event/control is normalized by its previous 24h variance. This
        # measures the event's increment over recent volatility, not raw level.
        def level(t):
            return squared.loc[t-pd.Timedelta(days=1):t-pd.Timedelta(minutes=31)].mean()
        scale=level(time)
        if not np.isfinite(scale) or scale<=0:
            continue
        for lo,hi in WINDOWS:
            def observation(t):
                z=squared.loc[t+pd.Timedelta(minutes=lo):t+pd.Timedelta(minutes=hi-1)]
                if z.notna().sum()<.8*(hi-lo):
                    return np.nan
                return z.mean()/max(level(t),1e-14)
            event=observation(time)
            control=np.nanmean([observation(t) for t in controls])
            if np.isfinite(event) and np.isfinite(control) and control>0:
                rows.append({'asset':asset,'time':time,'events':types,'lo':lo,'hi':hi,'event_normalized_var':event,'control_normalized_var':control,'ratio':event/control,'controls':len(controls)})
    return pd.DataFrame(rows)


def main():
    os.chdir(ROOT)
    with ProcessPoolExecutor(max_workers=2) as pool:
        frames=list(pool.map(asset_effects,CRYPTO+OTHER))
    df=pd.concat(frames,ignore_index=True)
    out=ROOT/'agent/results/events'
    out.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out/'matched_events.parquet',index=False)
    summary=df.groupby(['asset','events','lo','hi']).agg(n=('ratio','count'),median_ratio=('ratio','median'),mean_ratio=('ratio','mean'))
    summary.to_csv(out/'matched_summary.csv')
    print('Saved',len(df),'event/control measurements')
    print(summary.loc[(slice(None),slice(None),0,1),:].round(3).to_string())

if __name__=='__main__':
    main()
