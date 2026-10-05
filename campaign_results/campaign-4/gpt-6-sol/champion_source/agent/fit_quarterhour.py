"""Fit weekday quarter-hour variance profiles for the COM_EQU experiment."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.preparation.config import default_store_root


AGENT_DIR = Path(__file__).resolve().parent
ROOT = AGENT_DIR.parent
TRAIN_END = date(2026, 8, 31)
ASSETS = ("XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500")
OLD = json.loads((AGENT_DIR / "seasonality_halfhour_aug31.json").read_text())


def fit(asset: str) -> dict:
    root = default_store_root(asset)
    day = date(2026, 5, 18)
    paths = []
    while day <= TRAIN_END:
        path = root / f"date={day.isoformat()}.parquet"
        if path.exists():
            paths.append(path)
        day += timedelta(days=1)
    frame = pd.concat([pd.read_parquet(path, columns=["timestamp", "close"]) for path in paths], ignore_index=True)
    ts = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    close = frame["close"].to_numpy(dtype=float)
    valid = np.isfinite(close) & (close > 0)
    adjacent = valid[1:] & valid[:-1] & (np.diff(ts.to_numpy()) == np.timedelta64(1, "m"))
    returns = np.diff(np.log(np.where(valid, close, 1.0)))[adjacent]
    times = ts[1:][adjacent]
    cap = np.quantile(np.abs(returns), 0.999)
    sq = np.minimum(returns * returns, cap * cap)
    data = pd.DataFrame({
        "week": times.strftime("%G-%V"),
        "bin": times.hour * 4 + times.minute // 15,
        "weekend": times.dayofweek >= 5,
        "sq": sq,
    })
    weekly = data.groupby("week")["sq"].mean()
    data["normalized_sq"] = data["sq"].to_numpy() / data["week"].map(weekly).to_numpy()
    weekday = data.loc[~data["weekend"]]
    group = weekday.groupby("bin")["normalized_sq"].mean().reindex(range(96)).fillna(1.0).to_numpy()
    smooth = (np.roll(group, 1) + 2 * group + np.roll(group, -1)) / 4
    profile = 0.85 * smooth + 0.15 * np.mean(smooth)
    target = float(np.mean(OLD[asset]["weekday"]))
    profile *= target / float(np.mean(profile))
    peak = int(np.argmax(profile))
    print(asset, "valid minutes", len(returns), "peak UTC", f"{peak // 4:02d}:{15 * (peak % 4):02d}",
          "weekday range", round(float(np.min(profile)), 3), round(float(np.max(profile)), 3))
    return {
        "weekday": profile.round(5).tolist(),
        "weekend": OLD[asset]["weekend"],
        "train_end": TRAIN_END.isoformat(),
        "weekday_bin_minutes": 15,
    }


if __name__ == "__main__":
    os.chdir(ROOT)
    fitted = {asset: fit(asset) for asset in ASSETS}
    (AGENT_DIR / "seasonality_quarterhour_aug31.json").write_text(json.dumps(fitted, indent=2) + "\n")
