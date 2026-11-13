"""EDA 3: minute-of-hour effects (crypto), release-minute sizes, zero-return share by day type."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research")
from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
for a in ["BTC", "SOL", "HYPE"]:
    df = load_asset(a); r = np.log(df["close"]).diff()
    lvl = (r**2).rolling(1440*3, min_periods=1440, center=True).mean()
    u = (r**2 / lvl).clip(upper=100)
    g = u.groupby(u.index.minute).mean(); g = g / g.mean()
    print(a, "minute-of-hour var mult:", np.round(g.to_numpy(), 2))
    ny = u.index.tz_convert("America/New_York")
    wd = ny.day_of_week < 5
    sel = wd & (ny.hour == 9) & (ny.minute >= 20) & (ny.minute < 45)
    g2 = u[sel].groupby(ny[sel].minute).mean(); print("   NY 09:20-09:44 weekday var mult:", np.round(g2.to_numpy(), 1))
    sel = wd & (ny.hour == 8) & (ny.minute >= 25) & (ny.minute < 45)
    g2 = u[sel].groupby(ny[sel].minute).mean(); print("   NY 08:25-08:44 weekday var mult:", np.round(g2.to_numpy(), 1))
for a in ["SP500", "NVDAX", "XAU"]:
    df = load_asset(a); r = np.log(df["close"]).diff()
    lvl = (r**2).rolling(1440*3, min_periods=1440, center=True).mean()
    u = (r**2 / lvl).clip(upper=200)
    ny = u.index.tz_convert("America/New_York"); wd = ny.day_of_week < 5
    for hh, m0, m1 in [(9, 20, 45), (8, 25, 45), (16, 0, 15), (15, 50, 60)]:
        sel = wd & (ny.hour == hh) & (ny.minute >= m0) & (ny.minute < m1)
        g2 = u[sel].groupby(ny[sel].minute).mean(); print(a, f"NY {hh}:{m0}-{m1} weekday var mult:", np.round(g2.to_numpy(), 1))
    z = (r == 0)
    last = z.index >= z.index[-1] - pd.Timedelta(days=90)
    dt = np.where(ny.day_of_week == 5, "Sat", np.where(ny.day_of_week == 6, "Sun", "Wkd"))
    print(a, "zero 1-min share last 90d by daytype:", z[last].groupby(dt[last]).mean().round(3).to_dict(),
          "| share of sum|r| by daytype:", (r.abs()[last].groupby(dt[last]).sum() / r.abs()[last].sum()).round(3).to_dict())
