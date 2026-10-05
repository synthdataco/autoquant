"""Fit pre-validation minute-volatility seasonality from the frozen price snapshot."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.preparation.config import default_store_root


ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = date(2026, 7, 31)
ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE", "XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500")


def fit(asset: str) -> dict:
    root = default_store_root(asset)
    start = date(2026, 1, 1) if asset in ASSETS[:5] else date(2026, 5, 18)
    paths = []
    day = start
    while day <= TRAIN_END:
        p = root / f"date={day.isoformat()}.parquet"
        if p.exists():
            paths.append(p)
        day += timedelta(days=1)
    frames = [pd.read_parquet(p, columns=["timestamp", "close"]) for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    close = frame["close"].to_numpy(dtype=float)
    ret = np.log(close[1:] / close[:-1])
    good = np.isfinite(ret) & (np.diff(ts.to_numpy()).astype("timedelta64[m]").astype(int) == 1)
    ret = ret[good]
    times = pd.DatetimeIndex(ts[1:][good])
    # Robustify the variance estimate against one-off outliers in an hour cell.
    cap = np.quantile(np.abs(ret), 0.999)
    sq = np.minimum(ret * ret, cap * cap)
    df = pd.DataFrame({"week": times.strftime("%G-%V"), "hour": times.hour, "weekend": times.dayofweek >= 5, "sq": sq})
    weekly = df.groupby("week")["sq"].mean()
    df["normalized_sq"] = df["sq"].to_numpy() / df["week"].map(weekly).to_numpy()
    # Smooth hourly cells to suppress noise, then shrink modestly toward the
    # corresponding weekday/weekend average.
    arr = np.ones((2, 24), dtype=float)
    for weekend in (0, 1):
        subset = df.loc[df["weekend"] == bool(weekend)]
        group = subset.groupby("hour")["normalized_sq"].mean().reindex(range(24)).fillna(1.0).to_numpy()
        smooth = (np.roll(group, 1) + 2 * group + np.roll(group, -1)) / 4
        arr[weekend] = 0.85 * smooth + 0.15 * np.mean(smooth)
    # Preserve the asset's mean variance, accounting for weekday frequency.
    arr /= (5 * np.mean(arr[0]) + 2 * np.mean(arr[1])) / 7
    print(asset, "minutes", len(ret), "weekday ratio", round(float(np.mean(arr[0])), 3),
          "hour range", round(float(np.min(arr[0])), 3), round(float(np.max(arr[0])), 3),
          "weekend ratio", round(float(np.mean(arr[1])), 3))
    return {"weekday": arr[0].round(5).tolist(), "weekend": arr[1].round(5).tolist(), "train_end": TRAIN_END.isoformat()}


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=date.fromisoformat, default=TRAIN_END)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "seasonality.json")
    args = parser.parse_args()
    TRAIN_END = args.train_end
    os.chdir(ROOT)
    fitted = {asset: fit(asset) for asset in ASSETS}
    args.output.write_text(json.dumps(fitted, indent=2) + "\n")
