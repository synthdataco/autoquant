"""Sunday reopen: size of the move around 18:00 NY (CME open) per Sunday vs. the surrounding minutes."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "research"); sys.path.insert(0, "agent")
from common import *
import core
np.set_printoptions(linewidth=220, precision=1, suppress=True)
for a in ["XAU", "WTIOIL", "SP500", "NVDAX"]:
    df = load_asset(a); c = df["close"]
    wall = core.ny_wall_minutes(df.index); mow = core.minute_of_week(wall)
    lr = np.log(c.to_numpy()); r = np.full(len(c), np.nan); r[1:] = np.diff(lr)
    for label, m0 in [("Sun 18:00", 6 * 1440 + 1080), ("Sun 20:00", 6 * 1440 + 1200), ("Mon 18:00", 1080)]:
        idx = np.flatnonzero(mow == m0)
        idx = idx[(idx > 100) & (idx < len(c) - 100)]
        # abs 1-min returns (bps) by minute offset -3..+10, mean over weeks; and the first 5-min move distribution
        offs = np.arange(-3, 11)
        tab = np.array([[abs(r[i + o]) * 1e4 for o in offs] for i in idx])
        first5 = np.array([abs(lr[i + 4] - lr[i - 1]) * 1e4 for i in idx])
        pre30 = np.array([np.nanmean(np.abs(r[i - 30:i])) * 1e4 for i in idx])
        print(a, label, "n=", len(idx), "mean |r1| bps by offset -3..10:", np.nanmean(tab, 0))
        print("     first-5min |move| bps quantiles 50/75/90/max:", np.nanquantile(first5, [0.5, 0.75, 0.9, 1.0]), " mean pre-30min |r1|:", np.nanmean(pre30).round(2))
