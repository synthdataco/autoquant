"""EDA 1: variance ratios, zero-return mass, tails, over the last 120 days of each asset."""
import sys; sys.path.insert(0, "research")
from common import *
pd.set_option("display.width", 250)
rows = []
for a in ALL_ASSETS:
    df = load_asset(a)
    c = df["close"].iloc[-120*1440:]
    lp = np.log(c.to_numpy())
    r1 = np.diff(lp)
    ok = np.isfinite(r1)
    r1c = r1[ok]
    def agg(k):
        x = lp[::k]; d = np.diff(x); return d[np.isfinite(d)]
    r5, r30, r180, r1440 = agg(5), agg(30), agg(180), agg(1440)
    v1 = np.mean(r1c**2)
    row = dict(asset=a,
        sd1_bps=np.sqrt(v1)*1e4, sd5_bps=np.sqrt(np.mean(r5**2))*1e4,
        zero1=np.mean(r1c==0), zero5=np.mean(r5==0),
        ac1_1m=np.corrcoef(r1c[:-1], r1c[1:])[0,1],
        VR5=np.mean(r5**2)/(5*v1), VR30=np.mean(r30**2)/(30*v1), VR180=np.mean(r180**2)/(180*v1), VR1440=np.mean(r1440**2)/(1440*v1),
        VR30_5=np.mean(r30**2)/(6*np.mean(r5**2)), VR180_5=np.mean(r180**2)/(36*np.mean(r5**2)),
        ac1_5m=np.corrcoef(r5[:-1], r5[1:])[0,1],
        kurt5=np.mean(r5**4)/np.mean(r5**2)**2,
        mad_ratio5=np.mean(np.abs(r5))/np.sqrt(np.mean(r5**2)),  # 0.798 for gaussian
        tc0=np.mean(df["trade_count"].iloc[-120*1440:]==0),
    )
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
