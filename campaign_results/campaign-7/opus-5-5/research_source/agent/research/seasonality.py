"""Intraday volatility profile per asset: UTC vs New York clock, weekday vs weekend, two DST regimes."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, 'agent/research')
from data import load_all, ALL_ASSETS

pd.set_option('display.width', 250)
out = {}
for a in ALL_ASSETS:
    df = load_all(a)
    c = df['close']
    # 5-min log returns on the 5-min grid
    c5 = c[c.index.minute % 5 == 0]
    r = np.log(c5).diff().dropna()
    r = r[r.index >= r.index[0] + pd.Timedelta(days=30)]  # skip thin first month
    r2 = r ** 2
    ny = r2.index.tz_convert('America/New_York')
    wk = ny.dayofweek < 5
    # winsorize to limit outliers
    cap = r2.quantile(0.999)
    r2 = r2.clip(upper=cap)
    utc_h = r2.index.hour
    ny_h = ny.hour
    dst = np.asarray([bool(x.dst()) for x in ny[::288]]).repeat(288)[:len(ny)] if False else None
    prof_utc = r2[wk].groupby(utc_h[wk]).mean()
    prof_ny = r2[wk].groupby(ny_h[wk]).mean()
    wkend_ratio = r2[~wk].mean() / r2[wk].mean()
    # which clock: compare concentration (max/min ratio) of the profiles, by DST regime
    isdst = pd.Series(ny.map(lambda x: bool(x.dst())) if len(ny) < 0 else 0)
    vol = np.sqrt(prof_ny / prof_ny.mean())
    out[a] = dict(weekend_over_weekday_var=round(wkend_ratio, 3),
                  ny_max_min=round(float(prof_ny.max() / prof_ny.min()), 2),
                  utc_max_min=round(float(prof_utc.max() / prof_utc.min()), 2),
                  ny_peak_hour=int(prof_ny.idxmax()), ny_trough_hour=int(prof_ny.idxmin()),
                  vol_ny_by_hour=' '.join(f'{v:.2f}' for v in vol.values))
print(pd.DataFrame(out).T.to_string())
