"""Vol-level forecast study (no simulation): QLIKE of candidate deseasonalised-variance forecasts
against realised deseasonalised variance over the next H minutes, on an hourly grid."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "agent")
from data import load_asset

SEASON = np.load("agent/artifacts/season_2026-08-01.npz")


def zseries(asset):
    c = load_asset(asset)["close"]
    lp = np.log(c).ffill()
    r2 = lp.diff() ** 2
    ny = r2.index.tz_convert("America/New_York")
    b = (ny.dayofweek * 288 + ny.hour * 12 + ny.minute // 5).to_numpy()
    return r2 / SEASON[asset][b]


def features(z, hls=(15, 60, 240, 720, 1440, 4320)):
    f = {f"e{h}": z.ewm(halflife=h, ignore_na=True, min_periods=h).mean() for h in hls}
    f["l7d"] = z.rolling(7 * 1440, min_periods=3 * 1440).mean()
    return pd.DataFrame(f)


def targets(z, Hs=(60, 1440)):
    return pd.DataFrame({f"y{H}": z[::-1].rolling(H, min_periods=H // 2).mean()[::-1].shift(-1) for H in Hs})


def qlike(y, f):
    r = y / f
    return np.mean(r - np.log(r) - 1)


if __name__ == "__main__":
    out = []
    for a in ["BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]:
        z = zseries(a)
        F = features(z); Y = targets(z)
        D = pd.concat([F, Y], axis=1).iloc[::60].dropna()
        D = D[D.index >= "2026-02-01"]
        for H in (60, 1440):
            y = D[f"y{H}"]
            row = {"asset": a, "H": H}
            for c in F.columns:
                row[c] = qlike(y, D[c])
            out.append(row)
    df = pd.DataFrame(out).set_index(["H", "asset"])
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())
