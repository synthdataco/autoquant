"""Fit regularized day-of-week factors on calendar-adjusted COM_EQU variance."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.preparation.config import default_store_root

from fiveminute_calendar import _factors


AGENT_DIR = Path(__file__).resolve().parent
ROOT = AGENT_DIR.parent
TRAIN_END = date(2026, 8, 31)
ASSETS = ("XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500")


def fit(asset: str) -> list[float]:
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
    adjusted_sq = np.minimum(returns * returns, cap * cap) / _factors(asset, times)
    data = pd.DataFrame({"week": times.strftime("%G-%V"), "day": times.dayofweek, "adjusted_sq": adjusted_sq})
    weekly = data.groupby("week")["adjusted_sq"].mean()
    data["normalized_sq"] = data["adjusted_sq"].to_numpy() / data["week"].map(weekly).to_numpy()
    factors = data.groupby("day")["normalized_sq"].mean().reindex(range(7)).fillna(1.0).to_numpy().copy()
    factors[:5] /= factors[:5].mean()
    factors[5:] /= factors[5:].mean()
    factors = 0.5 + 0.5 * factors
    print(asset, "day factors", np.round(factors, 3).tolist(), flush=True)
    return factors.round(5).tolist()


if __name__ == "__main__":
    os.chdir(ROOT)
    output = {asset: {"day_factors": fit(asset), "train_end": TRAIN_END.isoformat()} for asset in ASSETS}
    (AGENT_DIR / "seasonality_dayofweek_aug31.json").write_text(json.dumps(output, indent=2) + "\n")
