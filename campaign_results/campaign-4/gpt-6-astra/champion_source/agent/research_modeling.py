"""Seasonal, adaptive return simulation; all runtime calibration uses past context."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

_SEASON=json.loads(Path(__file__).with_name(os.environ.get('SEASON_FILE','seasonality.json')).read_text())
_HAR_PATH=Path(__file__).with_name(os.environ.get('HAR_FILE','har.json'))
_HAR=json.loads(_HAR_PATH.read_text()) if _HAR_PATH.exists() else {}
_DEPENDENCE=json.loads(Path(__file__).with_name('dependence.json').read_text()) if Path(__file__).with_name('dependence.json').exists() else {}
_MEAN=json.loads(Path(__file__).with_name('mean.json').read_text()) if Path(__file__).with_name('mean.json').exists() else {}
# Environment switches exist for controlled ablations only; defaults define serving model.
def simulate(asset,start_time,time_increment=300,time_length=86400,num_simulations=1000,context_prices=None):
    seed=int(os.environ.get('SYNTH_BENCHMARK_SEED','42'))
    rng=np.random.default_rng(seed)
    mode=os.environ.get('MODEL_MODE','seasonal')
    scale=float(os.environ.get('MODEL_SCALE','1'))
    tail=float(os.environ.get('MODEL_TAIL','0'))
    mixture=float(os.environ.get('MODEL_MIX','0'))
    n=time_length//time_increment
    raw=context_prices['close'].astype(float)
    raw=raw.where(raw>0)
    valid=raw.dropna()
    if valid.empty:
        raise ValueError('context_prices must contain at least one positive observed close')
    last=float(valid.iloc[-1])
    if os.environ.get('MODEL_COMPLETE_BARS')=='1' and time_increment>60:
        boundary=(raw.index[-1]+pd.Timedelta(minutes=1)).floor(f'{time_increment}s')
        raw=raw.loc[raw.index<boundary]
    # Score the distribution on the serving grid, avoiding excess microstructure noise.
    close=raw.resample(f'{time_increment}s').last().dropna()
    r=np.diff(np.log(close.to_numpy()))
    elapsed=np.diff(close.index.asi8)/1e9
    bins=_SEASON.get('bins',24)
    shape=np.asarray(_SEASON['tables'].get(asset,[1.]*(7*bins)))
    def season_index(index):
        return index.dayofweek*bins+(index.hour*60+index.minute)//(1440//bins)
    seasonal=mode!='constant'
    # A frozen historical prior can only be used after its fit cutoff.
    if pd.Timestamp(start_time)<=pd.Timestamp(_SEASON['fit_end'],tz='UTC')+pd.Timedelta(days=1):
        seasonal=False
    if not seasonal:shape=np.ones(7*bins)
    idx=close.index[1:]
    histshape=shape[season_index(idx)]
    good=np.isfinite(r)&(elapsed>0)
    # Dividing by elapsed time handles thin observed series without filling prices.
    variance=r[good]**2/(np.maximum(elapsed[good],time_increment)/time_increment)/histshape[good]
    age=(pd.Timestamp(start_time)-idx[good]).total_seconds().to_numpy()/3600
    base=float(np.mean(variance)) if len(variance)>10 else 1e-6
    if mode in ('adaptive','adaptive_fast','absolute','absolute_fast'):
        half=6 if time_length==3600 else 24
        if mode in ('adaptive_fast','absolute_fast'):half/=3
        w=np.exp(-np.log(2)*age/half)
        if mode.startswith('absolute'):
            absolute=np.sqrt(variance)*np.sqrt(np.pi/2)
            base=float(np.mean(absolute))**2
            recent=float(np.sum(w*absolute)/np.sum(w))**2
        else:
            recent=float(np.sum(w*variance)/np.sum(w))
        # Revert a recent state toward weekly variance over the forecast horizon.
        future=np.arange(1,n+1)*time_increment/3600
        decay=np.exp(-np.log(2)*future/(12 if time_length==3600 else 24))
        level=.25*base+.75*(base+(recent-base)*decay)
    else:level=np.full(n,base)
    har_available=pd.Timestamp(start_time)>pd.Timestamp(_HAR.get('fit_end','2099-01-01'),tz='UTC')+pd.Timedelta(days=1)
    if mode in ('har','gbm','gbm_blend','ohlcv','ohlcv_blend') and seasonal and har_available and f'{asset}_{time_increment}' in _HAR.get('models',{}):
        grid=raw.resample(f'{time_increment}s').last()
        normalizer=np.sqrt(shape[season_index(grid.index)])
        abs_returns=np.log(grid).diff().abs()/normalizer
        features=np.array([abs_returns.iloc[-max(1,int(h*3600/time_increment)):].mean() for h in [.25,1,6,24,72,168]])
        record=_HAR['models'][f'{asset}_{time_increment}']
        # A thin venue can have no observed return in a short feature window.
        # Substitute the longer *observed* scale, never fabricate missing prices.
        fallback=float(np.nanmean(abs_returns))
        if not np.isfinite(fallback) or fallback<=0:
            fallback=float(np.exp(record['center'][-1]))
        features=np.where(np.isfinite(features)&(features>0),features,fallback)
        x=(np.log(np.maximum(features,1e-10))-record['center'])/record['sd']
        forecast=np.exp(np.dot(np.r_[1.,x],record['beta']))*record['correction']
        level=np.full(n,(forecast*np.sqrt(np.pi/2))**2)
        nn_weight=float(os.environ.get('MODEL_NN','0'))
        if nn_weight:
            from vol_features import features as neural_features
            global _NN
            if '_NN' not in globals():
                _NN=json.loads(Path(__file__).with_name('nn.json').read_text())
            frame,_,_=neural_features(raw,asset,time_increment,_SEASON)
            key='crypto1' if time_length==3600 else 'crypto24' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'com24'
            fitted=_NN['models'][key]
            xx=np.clip((frame.iloc[-1].to_numpy()-fitted['center'])/fitted['sd'],-5,5)
            xx=np.nan_to_num(xx)
            predictions=[]
            for net in fitted['nets']:
                yy=xx
                for layer in ['0','2']:
                    yy=np.asarray(net[layer+'.weight'])@yy+net[layer+'.bias']
                    yy=yy/(1+np.exp(-np.clip(yy,-40,40)))
                yy=np.asarray(net['4.weight'])@yy+net['4.bias']
                predictions.append(.3*np.tanh(yy[0]))
            correction=float(np.mean(predictions))
            level*=np.exp(2*nn_weight*correction)
        term_weight=float(os.environ.get('MODEL_TERM','0'))
        if term_weight:
            global _TERM
            if '_TERM' not in globals():
                _TERM=json.loads(Path(__file__).with_name('term.json').read_text())
            fitted=_TERM['models'][f'{asset}_{time_increment}']
            block_forecasts=[]
            for fit in fitted['fits']:
                xx=(np.log(np.maximum(features,1e-10))-fit['center'])/fit['sd']
                block_forecasts.append(np.exp(np.dot(np.r_[1.,xx],fit['beta']))*fit['correction'])
            edges=np.asarray(fitted['edges'])
            # Piecewise expected scale; this interpolates no observed prices.
            indices=np.minimum(np.searchsorted(edges[1:],np.arange(n)*time_increment/60,side='right'),len(block_forecasts)-1)
            term=np.asarray(block_forecasts)[indices]
            blended=forecast**(1-term_weight)*term**term_weight
            level=(blended*np.sqrt(np.pi/2))**2
        if mode in ('ohlcv','ohlcv_blend'):
            from ohlcv_features import features as ohlcv_features
            global _OHLCV
            if '_OHLCV' not in globals():
                _OHLCV=json.loads(Path(__file__).with_name('ohlcv.json').read_text())
            x,_=ohlcv_features(context_prices,time_increment,_SEASON,asset)
            fitted=_OHLCV['models'][f'{asset}_{time_increment}']
            value=x.iloc[-1].to_numpy()
            value=np.where(np.isfinite(value),value,fitted['center'])
            xx=(value-fitted['center'])/fitted['sd']
            extended=np.exp(np.dot(np.r_[1.,xx],fitted['beta']))*fitted['correction']
            if mode=='ohlcv_blend':extended=np.sqrt(extended*forecast)
            level=np.full(n,(extended*np.sqrt(np.pi/2))**2)
        if mode in ('gbm','gbm_blend'):
            import pickle
            from vol_features import features
            global _GBM
            if '_GBM' not in globals():
                with Path(__file__).with_name('gbm.pkl').open('rb') as f:
                    _GBM=pickle.load(f)
            x,_,baseline=features(raw,asset,time_increment,_SEASON)
            key='crypto1' if time_length==3600 else 'crypto24' if asset in ['BTC','ETH','SOL','XRP','HYPE'] else 'com24'
            fitted=_GBM['models'][key]
            prediction=float(fitted['model'].predict(x.iloc[[-1]].to_numpy())[0])
            gbm_forecast=float(baseline.iloc[-1])*np.exp(prediction)*fitted['correction']
            if mode=='gbm_blend':gbm_forecast=np.sqrt(gbm_forecast*forecast)
            level=np.full(n,(gbm_forecast*np.sqrt(np.pi/2))**2)
    future_times=pd.date_range(pd.Timestamp(start_time),periods=n,freq=f'{time_increment}s')
    future_shape=shape[season_index(future_times)]
    sigma=np.sqrt(np.maximum(level*future_shape,1e-14))*scale
    event_scale=float(os.environ.get('MODEL_FOMC','1'))
    if event_scale!=1 and asset in ['BTC','ETH','SOL','XRP','HYPE']:
        # Scheduled meeting dates, not policy decisions or realized future outcomes.
        dates={'2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29','2026-09-16'}
        local=future_times.tz_convert('America/New_York')
        event=np.array([(d.isoformat() in dates and h==14) for d,h in zip(local.date,local.hour)])
        sigma=sigma*np.where(event,event_scale,1.)
    half=(num_simulations+1)//2
    z=rng.standard_normal((half,n))
    if tail>2:z=rng.standard_t(tail,(half,n))*np.sqrt((tail-2)/tail)
    if os.environ.get('MODEL_INNOVATIONS')=='empirical':
        residual_grid=raw.resample(f'{time_increment}s').last()
        residual_returns=np.log(residual_grid).diff()/np.sqrt(shape[season_index(residual_grid.index)])
        local=residual_returns.abs().ewm(halflife=max(1,21600/time_increment),min_periods=20).mean().shift(1)
        residual=(residual_returns/local).replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
        if len(residual)>100:
            residual=np.clip(residual,-20,20)
            residual=residual/max(float(np.mean(np.abs(residual))),1e-10)*np.sqrt(2/np.pi)
            z=rng.choice(residual,size=(half,n))
            block=int(os.environ.get('MODEL_BLOCK','1'))
            if block>1:
                starts=rng.integers(0,len(residual)-block+1,size=(half,(n+block-1)//block))
                z=residual[starts[:,:,None]+np.arange(block)].reshape(half,-1)[:,:n]
    rho=float(os.environ.get('MODEL_RHO','0'))
    if os.environ.get('MODEL_DEPENDENCE')=='fitted':
        rho=_DEPENDENCE.get('parameters',{}).get(f'{asset}_{time_increment}',0.)
    if rho:
        # MA(1) preserves one-step variance but changes multistep variance.
        theta=2*rho/(1+np.sqrt(1-4*rho*rho))
        previous=rng.standard_normal((half,1))
        z=(z+theta*np.concatenate([previous,z[:,:-1]],axis=1))/np.sqrt(1+theta*theta)
    # Antithetic paths reduce irrelevant Monte Carlo drift and CRPS noise.
    z=np.concatenate([z,-z],axis=0)[:num_simulations]
    if mixture:
        factors=np.exp(mixture*rng.standard_normal((half,1))-.5*mixture**2)
        factors=np.concatenate([factors,factors],axis=0)[:num_simulations]
        z*=factors
    changes=z*sigma
    drift=float(os.environ.get('MODEL_DRIFT','0'))
    if drift and seasonal and f'{asset}_{time_increment}' in _MEAN.get('models',{}):
        grid=raw.resample(f'{time_increment}s').last()
        lg=np.log(grid)
        local_scale=float(lg.diff().abs().iloc[-max(1,86400//time_increment):].mean())*np.sqrt(np.pi/2)
        features=[]
        for h in [.25,1,6,24,72]:
            lag=max(1,int(h*3600/time_increment))
            past=lg.iloc[max(0,len(lg)-lag-1)]
            val=(lg.iloc[-1]-past)/max(local_scale*np.sqrt(lag),1e-10)
            features.append(float(np.clip(val,-4,4)) if np.isfinite(val) else 0.)
        normalized=float(np.dot(features,_MEAN['models'][f'{asset}_{time_increment}']))
        total_mean=float(np.clip(normalized,-.3,.3))*local_scale*np.sqrt(n)*drift
        changes+=total_mean/n
    logs=np.concatenate([np.zeros((num_simulations,1)),np.cumsum(changes,axis=1)],axis=1)
    paths=last*np.exp(logs)
    return (start_time,time_increment,*paths.tolist())
