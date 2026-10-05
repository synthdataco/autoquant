"""RESEARCH ONLY (uses the future!): modeling.py with the vol LEVEL replaced by the realised
deseasonalised variance over the prompt's horizon. Bounds what a perfect level forecast buys."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import modeling as M  # noqa: E402
from data import load_asset  # noqa: E402

PARAMS = M.PARAMS
PARAMS["oracle_mode"] = "off"   # off / flat: one level for the whole horizon; "buckets": per HAR bucket
_orig = M._har_level


def simulate(asset, start_time, time_increment=300, time_length=86400, num_simulations=1000, context_prices=None):
    if PARAMS["oracle_mode"] == "off":
        return M.simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices)
    t0 = pd.Timestamp(start_time)
    c = load_asset(asset)["close"]
    lp = np.log(c.loc[t0 - pd.Timedelta(days=7, minutes=5): t0 + pd.Timedelta(seconds=time_length + 120)]).ffill()
    r2 = lp.diff() ** 2
    prof = M._season(asset)
    z = (r2 / M._minute_profile(prof, r2.index))
    fut = z.loc[t0 + pd.Timedelta(minutes=1): t0 + pd.Timedelta(seconds=time_length)]

    def lvl(zc, har, h):
        if PARAMS["oracle_mode"] == "flat":
            return np.full(len(h), np.nanmean(fut.to_numpy()))
        # piecewise: realised mean over each HAR bucket
        spec, _ = har
        out = np.empty(len(h))
        fz = fut.to_numpy()
        for a, b in spec["buckets"]:
            m = (h - 3 > a) & (h - 3 <= b)
            if m.any():
                out[m] = np.nanmean(fz[a:b]) if np.isfinite(fz[a:b]).any() else np.nanmean(fz)
        return out
    M._har_level = lvl
    try:
        return M.simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices)
    finally:
        M._har_level = _orig
