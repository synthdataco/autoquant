"""Relative E|r|/sqrt(v) by hour of the NY week on a validation window (1.00 = average width; >1 = too narrow there)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "research"); sys.path.insert(0, "agent")
import numpy as np, torch, core, pandas as pd
from pathlib import Path
import train as T
pdir, ds, tag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
vs, ve = T.minute_stamp(sys.argv[5]), T.minute_stamp(sys.argv[6])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
for a in sys.argv[4].split(","):
    path = ds / f"{a}_{tag}.npz"
    z0 = np.load(path); sel = (z0["t0"] >= vs) & (z0["t0"] < ve)
    d = T.Data(path, tag, lambda t: (t >= vs) & (t < ve))
    par = json.loads((pdir / f"model_{a}_{tag}.json").read_text())
    m = T.Model(tag).to(T.DEV); m.load_raw(par["raw"])
    with torch.no_grad(): v = m.variance(d, torch.arange(d.n, device=T.DEV)).cpu().numpy()
    inc = T.CFG[tag]["inc"]; H = v.shape[1]
    r = d.ret[1].cpu().numpy() / 1e4
    t0 = z0["t0"][sel]
    end_min = t0[:, None] + (np.arange(H)[None, :] + 1) * inc            # utc minute label of each step's last minute
    wall = core.ny_wall_minutes(pd.DatetimeIndex(end_min.ravel().astype("datetime64[m]")).tz_localize("UTC")).reshape(end_min.shape)
    how = (core.minute_of_week(wall) // 60)
    z = np.abs(r) / np.sqrt(v); base = z.mean()
    tab = np.array([z[how == k].mean() / base if (how == k).sum() > 200 else np.nan for k in range(168)]).reshape(7, 24)
    print(a, tag, "rows Mon..Sun, cols NY hour 0..23"); print(tab)
