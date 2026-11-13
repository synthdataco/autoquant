"""EDA 2: hourly volatility profile (weekdays), NY clock vs UTC clock, per DST regime."""
import sys; sys.path.insert(0, "research")
from common import *
pd.set_option("display.width", 300)
def prof(a, tz, lo, hi, weekend=False):
    df = load_asset(a)
    c = df["close"].loc[lo:hi]
    r = np.log(c).diff()
    idx = r.index.tz_convert(tz)
    wk = idx.dayofweek >= 5
    sel = wk if weekend else ~wk
    x = (r[sel]**2)
    g = x.groupby(idx[sel].hour).mean()
    return np.sqrt(g / x.mean())
# regimes: winter (2025-11-21..2026-03-07, NY=UTC-5), summer (2026-03-29..2026-10-24, NY=UTC-4, London=UTC+1)
for a in ["SP500", "NVDAX", "AAPLX", "XAU", "WTIOIL", "BTC", "ETH", "HYPE"]:
    first = load_asset(a).index[0]
    out = {}
    if first < pd.Timestamp("2026-01-10", tz="UTC"):
        out["win_UTC"] = prof(a, "UTC", "2025-11-21", "2026-03-07")
    out["sum_UTC"] = prof(a, "UTC", "2026-03-29", "2026-10-24")
    if first < pd.Timestamp("2026-01-10", tz="UTC"):
        out["win_NY"] = prof(a, "America/New_York", "2025-11-21", "2026-03-07")
    out["sum_NY"] = prof(a, "America/New_York", "2026-03-29", "2026-10-24")
    out["sum_WE_NY"] = prof(a, "America/New_York", "2026-03-29", "2026-10-24", weekend=True)
    t = pd.DataFrame(out).T.round(2)
    print(a); print(t.to_string())
