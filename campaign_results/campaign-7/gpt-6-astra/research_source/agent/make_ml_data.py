"""Causal feature/forward-label pairs for a volatility-only learning experiment."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from explore import load_asset
from ml_features import ASSETS,NAMES,vector_features
from seasonal_models import get_profile
from run_suite import ROOT, CRYPTO


def future_mean(a,n):
    return pd.Series(a).shift(-1).iloc[::-1].rolling(n,min_periods=max(2,int(.98*n))).mean().iloc[::-1].to_numpy()


def make_asset(asset,profile='original',prefix='ml',context_days=7,target_kind='normalized'):
    os.chdir(ROOT)
    frame=load_asset(asset)
    frames={86400:[],3600:[]}
    for month in pd.date_range(frame.index.min().floor('D').replace(day=1),pd.Timestamp('2026-11-01',tz='UTC'),freq='MS'):
        stop=min(month+pd.offsets.MonthBegin(1),pd.Timestamp('2026-11-09',tz='UTC'))
        chunk=frame.loc[(frame.index>=month-pd.Timedelta(days=context_days))&(frame.index<stop+pd.Timedelta(days=1))]
        if len(chunk)<3*1440:
            continue
        if profile=='original':
            table=get_profile(asset,month.strftime('%Y-%m-%d'),'America/New_York')
        else:
            from profile_models import table_for
            table=table_for(asset,month.strftime('%Y-%m-%d'),profile)
        for horizon in ([86400,3600] if asset in CRYPTO else [86400]):
            x,base,seas=vector_features(chunk,asset,table,horizon,context_days)
            logp=np.log(np.asarray(chunk.close,dtype=float))
            if horizon==3600:
                absolute=np.abs(np.diff(logp,prepend=np.nan))
                raw=absolute/np.sqrt(seas)
                if target_kind=='risk_weighted':
                    denominator=np.where(np.isfinite(absolute),np.sqrt(seas),np.nan)
                    target=future_mean(absolute,60)/future_mean(denominator,60)*np.sqrt(np.pi/2)
                else:
                    target=future_mean(raw,60)*np.sqrt(np.pi/2)
                lead=2
            else:
                d=np.r_[np.full(5,np.nan),logp[5:]-logp[:-5]]
                valid=pd.Series(np.isfinite(logp).astype(float)).rolling(6,min_periods=6).sum().to_numpy()==6
                d[~valid]=np.nan
                normal=pd.Series(seas).rolling(5,min_periods=5).sum().to_numpy()
                raw=np.abs(d)/np.sqrt(normal)
                target=np.full(len(chunk),np.nan)
                # Each target uses the exact five-minute phase of its start.
                phase=np.asarray(chunk.index.minute)%5
                for k in range(5):
                    select=np.flatnonzero(phase==k)
                    if target_kind=='risk_weighted':
                        absolute=np.abs(d[select])
                        denominator=np.where(np.isfinite(absolute),np.sqrt(normal[select]),np.nan)
                        target[select]=future_mean(absolute,288)/future_mean(denominator,288)*np.sqrt(np.pi/2)
                    else:
                        target[select]=future_mean(raw[select],288)*np.sqrt(np.pi/2)
                lead=3
            valid_count=pd.Series(np.isfinite(logp).astype(float)).rolling(context_days*1440,min_periods=1).sum().to_numpy()
            select=np.flatnonzero((chunk.index>=month)&(chunk.index<stop)&np.isin(chunk.index.minute,[7,37]))
            select=select[(select>=lead)&(valid_count[np.maximum(select-lead,0)]>=2*1440)]
            y=target[select]/base[select-lead]
            valid=np.isfinite(y)&(y>0)&np.isfinite(base[select-lead])
            select=select[valid]
            out=pd.DataFrame(x[select-lead],columns=NAMES)
            out['start_time']=chunk.index[select]
            out['label_end']=chunk.index[select]+pd.Timedelta(seconds=horizon)
            out['asset']=asset
            out['time_length']=horizon
            out['base_sigma']=base[select-lead]
            out['target_ratio']=target[select]/base[select-lead]
            frames[horizon].append(out)
    count=0
    for horizon,parts in frames.items():
        if parts:
            out=pd.concat(parts,ignore_index=True)
            out=out[out.label_end<pd.Timestamp('2026-11-09',tz='UTC')]
            out.to_parquet(ROOT/'agent/cache'/f'{prefix}_{asset}_{horizon}.parquet',index=False)
            count+=len(out)
    return asset,count


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',default='original')
    ap.add_argument('--prefix',default='ml')
    ap.add_argument('--context-days',type=int,default=7)
    ap.add_argument('--target-kind',choices=['normalized','risk_weighted'],default='normalized')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    (ROOT/'agent/cache').mkdir(parents=True,exist_ok=True)
    work=partial(make_asset,profile=args.profile,prefix=args.prefix,context_days=args.context_days,target_kind=args.target_kind)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for a,n in pool.map(work,ASSETS):
            print(a,n,'training/validation rows',flush=True)

if __name__=='__main__':
    main()
