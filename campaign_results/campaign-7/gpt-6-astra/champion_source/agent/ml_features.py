"""Identical causal feature definitions for volatility training and serving."""
from __future__ import annotations
import numpy as np
import pandas as pd

from seasonal_models import get_profile, seasonal_values
from vol_models import valid_returns, ewma

ASSETS=['BTC','ETH','SOL','XRP','HYPE','XAU','NVDAX','GOOGLX','TSLAX','AAPLX','WTIOIL','SPCX','SP500']
HALVES=[15,60,240,720,1440,4320]
CORE_NAMES=[f'{kind}_{h}' for kind in ('rms','abs') for h in HALVES]+['log_base']
EXTRA_NAMES=[f'range_{h}' for h in (60,720,4320)]+['volume_60_1440','volume_240_4320','trades_60_1440','trades_240_4320','zero_60','zero_720']+[f'signed_{w}' for w in (30,120,360,1440)]
TIME_NAMES=['hour_sin','hour_cos','week_sin','week_cos']
NAMES=CORE_NAMES+EXTRA_NAMES+TIME_NAMES+['asset_'+a for a in ASSETS]


def finite_ewm(x, half, length):
    """Vectorized exactly truncated, NaN-aware exponential average."""
    x=np.asarray(x,dtype=float)
    good=np.isfinite(x)
    alpha=-np.expm1(-np.log(2)/half)
    n=pd.Series(np.r_[0.,np.where(good,x,0.)]).ewm(alpha=alpha,adjust=False).mean().to_numpy(copy=True)[1:]
    d=pd.Series(np.r_[0.,good.astype(float)]).ewm(alpha=alpha,adjust=False).mean().to_numpy(copy=True)[1:]
    if len(x)>length:
        tail=np.exp2(-length/half)
        n[length:] -= tail*n[:-length].copy()
        d[length:] -= tail*d[:-length].copy()
    return np.divide(n,d,out=np.full_like(n,np.nan),where=d>1e-12)


def time_features(starts, asset):
    idx=pd.DatetimeIndex(starts).tz_convert('America/New_York')
    hour=np.asarray(idx.hour+idx.minute/60)
    week=np.asarray(idx.day_of_week)+hour/24
    time=np.column_stack([np.sin(hour*2*np.pi/24),np.cos(hour*2*np.pi/24),np.sin(week*2*np.pi/7),np.cos(week*2*np.pi/7)])
    onehot=np.zeros((len(idx),len(ASSETS)))
    if asset in ASSETS:
        onehot[:,ASSETS.index(asset)]=1
    return np.column_stack([time,onehot])


def _one_mean(x,age,half):
    good=np.isfinite(x)
    w=np.exp2(-age[good]/half)
    return np.sum(w*x[good])/np.sum(w) if len(w) and w.sum()>1e-15 else np.nan


def one_features(asset,start_time,time_length,context,table=None):
    last,r,age=valid_returns(context)
    if table is None:
        table=get_profile(asset,str(context.index[-1].date()),'America/New_York')
    seas=seasonal_values(context.index,table,'America/New_York')
    r=r/np.sqrt(seas[1:])
    rms=np.array([ewma(r,age,h) for h in HALVES])
    absolute=np.array([ewma(r,age,h,True) for h in HALVES])
    base=np.sqrt(.75*rms[1]+.25*rms[3]) if time_length<=3600 else np.sqrt(.75*rms[3]+.25*rms[5])
    base=max(base,1e-7)
    core=np.r_[.5*np.log(np.maximum(rms,1e-14)/base**2),.5*np.log(np.maximum(absolute,1e-14)/base**2),np.log(base)]
    high=np.asarray(context['high'],dtype=float)[1:]
    low=np.asarray(context['low'],dtype=float)[1:]
    ranges=(np.log(np.maximum(high,1e-12)/np.maximum(low,1e-12)))**2/(4*np.log(2))/seas[1:]
    ranges[(high<=0)|(low<=0)]=np.nan
    extras=[.5*np.log(max(_one_mean(ranges,age,h),1e-14)/base**2) for h in (60,720,4320)]
    for col in ['volume','trade_count']:
        x=np.asarray(context[col],dtype=float)[1:].copy()
        x[~np.isfinite(r)]=np.nan
        for fast,slow in [(60,1440),(240,4320)]:
            a,b=_one_mean(x,age,fast),_one_mean(x,age,slow)
            extras.append(np.log(max(a,1e-12)/max(b,1e-12)))
    zeros=np.where(np.isfinite(r),(r==0).astype(float),np.nan)
    extras += [_one_mean(zeros,age,h) for h in (60,720)]
    for w in (30,120,360,1440):
        mask=(age<w)&np.isfinite(r)
        extras.append(np.sum(r[mask])/(base*np.sqrt(w)))
    x=np.r_[core,extras,time_features([pd.Timestamp(start_time)],asset)[0]]
    return x.astype(np.float32),base


def vector_features(frame,asset,table,time_length,context_days=7):
    """Every row is a possible context endpoint; future labels are separate."""
    p=np.asarray(frame.close,dtype=float)
    r=np.diff(np.log(p),prepend=np.nan)
    seas=seasonal_values(frame.index,table,'America/New_York')
    r=r/np.sqrt(seas)
    length=context_days*1440-(3 if time_length>3600 else 2)
    rms=np.column_stack([finite_ewm(r*r,h,length) for h in HALVES])
    av=np.column_stack([finite_ewm(np.abs(r),h,length)*np.sqrt(np.pi/2) for h in HALVES])
    base=np.sqrt(.75*rms[:,1]+.25*rms[:,3]) if time_length<=3600 else np.sqrt(.75*rms[:,3]+.25*rms[:,5])
    base=np.maximum(base,1e-7)
    core=np.column_stack([.5*np.log(np.maximum(rms,1e-14)/base[:,None]**2),np.log(np.maximum(av,1e-7)/base[:,None]),np.log(base)])
    high=np.asarray(frame.high,dtype=float)
    low=np.asarray(frame.low,dtype=float)
    ranges=np.log(np.maximum(high,1e-12)/np.maximum(low,1e-12))**2/(4*np.log(2))/seas
    ranges[(high<=0)|(low<=0)]=np.nan
    extras=[.5*np.log(np.maximum(finite_ewm(ranges,h,length),1e-14)/base**2) for h in (60,720,4320)]
    for col in ['volume','trade_count']:
        x=np.asarray(frame[col],dtype=float).copy()
        x[~np.isfinite(r)]=np.nan
        for fast,slow in [(60,1440),(240,4320)]:
            a,b=finite_ewm(x,fast,length),finite_ewm(x,slow,length)
            extras.append(np.log(np.maximum(a,1e-12)/np.maximum(b,1e-12)))
    zeros=np.where(np.isfinite(r),(r==0).astype(float),np.nan)
    extras += [finite_ewm(zeros,h,length) for h in (60,720)]
    for w in (30,120,360,1440):
        sums=pd.Series(r).rolling(w,min_periods=1).sum().to_numpy()
        extras.append(sums/(base*np.sqrt(w)))
    start=frame.index+pd.Timedelta(minutes=3 if time_length>3600 else 2)
    out=np.column_stack([core,*extras,time_features(start,asset)])
    return out.astype(np.float32),base,seas
