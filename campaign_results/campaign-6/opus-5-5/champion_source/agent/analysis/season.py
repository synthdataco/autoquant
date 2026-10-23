import pandas as pd, numpy as np, glob, sys
from pathlib import Path
R=Path('/workspace/market_data/prices')
def load(a, start=None):
    fs=sorted(glob.glob(str(R/a/'1m'/'date=*.parquet')))
    df=pd.concat([pd.read_parquet(f,columns=['timestamp','close']) for f in fs])
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True); return df.set_index('timestamp').close
def main():
  for a in sys.argv[1:]:
      c=load(a)
      c5=c[c.index.minute%5==0]
      r=np.log(c5).diff()
      r=r[c5.index.to_series().diff()==pd.Timedelta('5min')].dropna() if False else r.dropna()
      r=r['2026-04-01':]
      lvl=(r**2).rolling('7D',center=True).mean() if False else (r**2).rolling(2016,center=True,min_periods=500).mean()
      z=(r**2/lvl).dropna()
      idx=z.index
      dow=idx.dayofweek; hod=idx.hour
      t=pd.DataFrame({'z':z.values,'dow':dow,'h':hod})
      print(a, 'dow:', t.groupby('dow').z.mean().round(2).tolist())
      wd=t[t.dow<5].groupby('h').z.mean(); we=t[t.dow>=5].groupby('h').z.mean()
      print('  wkday h:', ' '.join(f'{v:.1f}' for v in wd.values))
      print('  wkend h:', ' '.join(f'{v:.1f}' for v in we.values))

if __name__=='__main__': main()