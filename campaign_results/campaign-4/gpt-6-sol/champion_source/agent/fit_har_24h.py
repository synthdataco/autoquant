"""Fit a 24-hour HAR-style variance forecast from the frozen crypto snapshot."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.preparation.config import default_store_root


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ("BTC", "ETH", "SOL", "XRP", "HYPE")
TRAIN_START = date(2026, 1, 1)
TRAIN_END = date(2026, 7, 31)
WINDOWS = (60, 360, 1440, 10080)
SEASONALITY = json.loads((Path(__file__).parent / "seasonality.json").read_text())


def fit(asset: str) -> dict:
    root = default_store_root(asset)
    frames = []
    day = TRAIN_START
    while day <= TRAIN_END:
        frames.append(pd.read_parquet(root / f"date={day.isoformat()}.parquet", columns=["timestamp", "close"]))
        day += timedelta(days=1)
    frame = pd.concat(frames, ignore_index=True)
    times = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    closes = frame["close"].to_numpy(dtype=float)
    valid = np.isfinite(closes) & (closes > 0)
    factors = SEASONALITY[asset]
    hourly = np.where(
        times.dayofweek >= 5,
        np.asarray(factors["weekend"])[times.hour],
        np.asarray(factors["weekday"])[times.hour],
    )
    good = valid[1:] & valid[:-1] & (np.diff(times.to_numpy()) == np.timedelta64(1, "m"))
    log_close = np.log(np.where(valid, closes, 1.0))
    adjusted_sq = np.full(len(closes), np.nan)
    adjusted_sq[1:][good] = np.diff(log_close)[good] ** 2 / hourly[1:][good]
    observed = np.isfinite(adjusted_sq)
    vals = np.nan_to_num(adjusted_sq, nan=0.0)
    sums = np.concatenate(([0.0], np.cumsum(vals)))
    counts = np.concatenate(([0], np.cumsum(observed.astype(np.int64))))
    origins = np.arange(max(WINDOWS) - 1, len(closes) - 1441, 120)
    features = []
    valid_origins = np.ones(len(origins), dtype=bool)
    for width in WINDOWS:
        left = origins + 1 - width
        s = sums[origins + 1] - sums[left]
        c = counts[origins + 1] - counts[left]
        valid_origins &= c >= 0.8 * width
        features.append(s / np.maximum(c, 1))
    future_s = sums[origins + 1441] - sums[origins + 1]
    future_c = counts[origins + 1441] - counts[origins + 1]
    valid_origins &= future_c >= 1200
    target = future_s / np.maximum(future_c, 1)
    x = np.column_stack(features)[valid_origins]
    target = target[valid_origins]
    if len(target) < 1000:
        raise ValueError(f"Too few HAR training examples for {asset}: {len(target)}")
    x_log = np.log(np.maximum(x, 1e-12))
    y_log = np.log(np.maximum(target, 1e-12))
    x_mean = x_log.mean(axis=0)
    x_sd = x_log.std(axis=0)
    x_std = (x_log - x_mean) / x_sd
    design = np.column_stack((np.ones(len(x_std)), x_std))
    penalty = np.diag([0.0, 50.0, 50.0, 50.0, 50.0])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y_log)
    pred_median = np.exp(design @ coef)
    # Match the training mean variance; the regression itself predicts log variance.
    calibration = float(np.mean(target) / np.mean(pred_median))
    print(asset, "examples", len(target), "coef", np.round(coef, 3).tolist(),
          "calibration", round(calibration, 3), flush=True)
    return {
        "train_start": TRAIN_START.isoformat(), "train_end": TRAIN_END.isoformat(),
        "windows": list(WINDOWS), "x_mean": x_mean.tolist(), "x_sd": x_sd.tolist(),
        "coef": coef.tolist(), "calibration": calibration,
        "examples": len(target),
    }


if __name__ == "__main__":
    import os

    os.chdir(ROOT)
    artifact = {asset: fit(asset) for asset in ASSETS}
    (Path(__file__).parent / "har_24h.json").write_text(json.dumps(artifact, indent=2) + "\n")
