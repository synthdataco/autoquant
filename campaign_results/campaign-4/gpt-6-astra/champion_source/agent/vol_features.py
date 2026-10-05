"""Shared causal features for training and serving the nonlinear ablation."""
import numpy as np
import pandas as pd
ASSETS='BTC ETH SOL XRP HYPE XAU NVDAX GOOGLX TSLAX AAPLX WTIOIL SPCX SP500'.split()
HOURS=[.25,1,6,24,72,168]
def features(raw,asset,step,season):
    close=raw.resample(f'{step}s').last()
    r=np.log(close).diff()
    bins=season.get('bins',24)
    shape=np.array(season['tables'][asset])
    norm=np.sqrt(shape[close.index.dayofweek*bins+(close.index.hour*60+close.index.minute)//(1440//bins)])
    a=r.abs()/norm
    means={h:a.rolling(max(1,int(h*3600/step)),min_periods=1).mean() for h in HOURS}
    rms={h:np.sqrt((r/norm).pow(2).rolling(max(1,int(h*3600/step)),min_periods=1).mean()) for h in HOURS}
    base=means[168].clip(lower=1e-10)
    f={'base':np.log(base)}
    for h in HOURS:
        f[f'a{h}']=np.log((means[h]/base).clip(lower=1e-5,upper=1e5))
        f[f'rms{h}']=np.log((rms[h]/base).clip(lower=1e-5,upper=1e5))
    for h in [.25,1,6,24]:
        count=max(1,int(h*3600/step))
        f[f'signed{h}']=((np.log(close)-np.log(close.shift(count)))/(base*np.sqrt(count))).clip(-10,10)
    f['zeros']=r.eq(0).rolling(86400//step,min_periods=1).mean()
    f['coverage']=r.notna().rolling(86400//step,min_periods=1).mean()
    hour=close.index.hour+close.index.minute/60
    f['hour_sin']=np.sin(hour*np.pi/12)
    f['hour_cos']=np.cos(hour*np.pi/12)
    f['weekend']=(close.index.dayofweek>=5).astype(float)
    f['asset']=float(ASSETS.index(asset))
    return pd.DataFrame(f,index=close.index),a,base
