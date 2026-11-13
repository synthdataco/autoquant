"""Is there a UTC-fixed volatility component beyond the New-York-clock profile? Backfit log u = A[NY 15-min-of-week] + B[UTC hour, weekday/weekend]."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
import core
np.set_printoptions(linewidth=250, precision=2, suppress=True)
for a in sys.argv[1].split(","):
    df = load_asset(a); idx = df.index
    r = np.log(df["close"]).diff().abs()
    lvl = (r ** 2).rolling(10080, center=True, min_periods=2880).mean() ** 0.5
    u = (r / lvl).to_numpy()
    ok = np.isfinite(u)
    wall = core.ny_wall_minutes(idx); mowq = core.minute_of_week(wall) // 15          # 672 NY buckets
    um = core.utc_minutes(idx); uh = (um // 60) % 24; ud = ((um // 1440 + 3) % 7 >= 5).astype(int); ukey = ud * 24 + uh
    A = np.ones(672); B = np.ones(48)
    for it in range(6):
        num = np.bincount(mowq[ok], weights=(u / B[ukey])[ok], minlength=672); den = np.bincount(mowq[ok], minlength=672)
        A = num / np.maximum(den, 1); A /= A.mean()
        num = np.bincount(ukey[ok], weights=(u / A[mowq])[ok], minlength=48); den = np.bincount(ukey[ok], minlength=48)
        B = num / np.maximum(den, 1); B[:24] /= B[:24].mean(); B[24:] /= B[24:].mean()
    # explained share: mean |u| fit error with and without B, on the winter regime only (NY=UTC-5)
    winter = ok & ((wall - um) == -300)
    e0 = np.abs(np.log(np.maximum(u[winter], 1e-3) / 1.0)).mean()
    print(a, "B (sd-scale) weekday by UTC hour:", B[:24], "\n   weekend:", B[24:], " winter minutes share:", round(winter.sum() / ok.sum(), 2))
