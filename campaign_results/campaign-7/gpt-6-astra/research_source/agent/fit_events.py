"""Causal event-kernel estimates and leave-one-event-out stability diagnostics."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

from explore import load_asset
from ml_features import one_features
from ml_models import scale_adjustment
from seasonal_models import get_profile,seasonal_values
from synth_lib.benchmark.generate_predictions import context_slice
from run_suite import ROOT,CRYPTO,OTHER

# Kernels use only dates supplied by the calendar. Earnings have coarse session
# anchors, so their windows are wider than a precisely timed macro release.
MACRO_BINS=[(-5,0),(0,1),(1,5),(5,15),(15,60),(60,180)]
EARNINGS_BINS=[(-30,0),(0,30),(30,120),(120,240),(240,480)]


def event_type(names):
    names=set(names)
    for kind in ['EARNINGS','EIA_PETROLEUM','CPI','NFP','FOMC','PPI','PCE','GDP','RETAIL_SALES']:
        if kind in names:
            return 'GDP_PCE' if kind in ('GDP','PCE') else kind
    return None


def observe(asset):
    os.chdir(ROOT)
    frame=load_asset(asset)
    r=np.log(frame.close).diff()
    cal=pd.read_parquet(os.environ['SYNTH_BENCHMARK_CALENDAR'])
    cal=cal[(cal.scope=='all')|cal.scope.str.split(',').map(lambda x:asset in x)]
    cal=cal[(cal.time_utc<'2026-11-09')&(~cal.event.str.startswith('NYSE'))]
    records=[]
    for event_at,events in cal.groupby('time_utc'):
        kind=event_type(events.event)
        if kind is None or event_at<frame.index.min()+pd.Timedelta(days=14):
            continue
        bins=EARNINGS_BINS if kind=='EARNINGS' else MACRO_BINS
        issue_start=event_at-pd.Timedelta(minutes=35)
        for horizon in ([86400,3600] if asset in CRYPTO else [86400]):
            context=context_slice(frame,issue_start,horizon,7)
            _,base=one_features(asset,issue_start,horizon,context)
            adjustment=scale_adjustment(asset,issue_start,horizon,context,'vol',1.)
            table=get_profile(asset,str(context.index[-1].date()),'America/New_York')
            for lo,hi in bins:
                if event_at+pd.Timedelta(minutes=hi)>frame.index.max():
                    continue
                observed=r.loc[event_at+pd.Timedelta(minutes=lo):event_at+pd.Timedelta(minutes=hi-1)]
                if observed.notna().sum()<max(1,.8*(hi-lo)):
                    continue
                season=seasonal_values(observed.index,table,'America/New_York')
                standardized=np.asarray(observed)/(max(base*adjustment,1e-8)*np.sqrt(season))
                amplitude=float(np.nanmean(np.abs(standardized))*np.sqrt(np.pi/2))
                records.append({'asset':asset,'group':'crypto' if asset in CRYPTO else ('equity' if asset not in ['XAU','WTIOIL'] else asset),
                                'time_length':horizon,'event':kind,'time':event_at,'end':event_at+pd.Timedelta(minutes=hi),
                                'lo':lo,'hi':hi,'amplitude':amplitude,'observations':int(observed.notna().sum())})
    return pd.DataFrame(records)


def robust_center(values):
    # A winsorized mean of absolute returns is more stable than mean variance.
    a=np.asarray(values,dtype=float)
    if len(a)>=5:
        a=np.minimum(a,np.quantile(a,.9)*1.5)
    return float(np.mean(a)) if len(a) else 1.


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lookback',type=int,default=365)
    ap.add_argument('--output',default='events.json')
    args=ap.parse_args()
    os.chdir(ROOT)
    out=ROOT/'agent/results/events'
    out.mkdir(parents=True,exist_ok=True)
    observation_file=out/'causal_event_observations.parquet'
    if observation_file.exists():
        data=pd.read_parquet(observation_file)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            data=pd.concat(list(pool.map(observe,CRYPTO+OTHER)),ignore_index=True)
        data.to_parquet(observation_file,index=False)
    dates=['2026-03-01','2026-07-01','2026-08-01','2026-10-01','2026-11-01','2026-11-09']
    fits={}
    stability=[]
    for date in dates:
        if 'LANDING ORDER' in (ROOT/'agent/BUDGET.md').read_text():
            print('Landing order: stop fitting.',flush=True)
            return
        train=data[(data.end<pd.Timestamp(date,tz='UTC'))&(data.time>=pd.Timestamp(date,tz='UTC')-pd.Timedelta(days=args.lookback))]
        rows=[]
        for (group,horizon,event,lo,hi),bucket in train.groupby(['group','time_length','event','lo','hi']):
            # Assets share each macro surprise. Count dates, not asset rows.
            per_event=bucket.groupby('time').amplitude.mean()
            n=len(per_event)
            if n<3:
                continue
            center=robust_center(per_event.to_numpy())
            shrink=n/(n+4.)
            group_amplitude=max(1.,1+shrink*(center-1))
            for asset,sub in bucket.groupby('asset'):
                na=len(sub)
                individual=robust_center(sub.amplitude.to_numpy())
                asset_weight=na/(na+12.)
                amplitude=max(1.,asset_weight*individual+(1-asset_weight)*group_amplitude)
                amplitude=min(amplitude,15. if lo<5 else 5.)
                rows.append({'asset':asset,'time_length':int(horizon),'event':event,'lo':int(lo),'hi':int(hi),
                             'amplitude':amplitude,'n_dates':n,'n_asset_dates':na})
            if date=='2026-11-09':
                jackknife=[robust_center(per_event.drop(t).to_numpy()) for t in per_event.index]
                stability.append({'group':group,'time_length':horizon,'event':event,'lo':lo,'hi':hi,'n_dates':n,
                                  'mean_absolute_scale_ratio':center,'loo_min':min(jackknife),'loo_max':max(jackknife)})
        fits[date]=rows
    artifact={'reference':'causal ml_vol Gaussian background','lookback_days':args.lookback,'fits':fits}
    dest=ROOT/'agent/artifacts'/args.output
    dest.write_text(json.dumps(artifact,indent=2)+'\n')
    suffix='' if args.lookback==365 else f'_{args.lookback}'
    pd.DataFrame(stability).to_csv(out/f'leave_one_event_out{suffix}.csv',index=False)
    print('Saved',len(data),'event observations;',sum(map(len,fits.values())),'causal event coefficients')
    print(pd.DataFrame(stability).query('lo==0').round(3).to_string(index=False))

if __name__=='__main__':
    main()
