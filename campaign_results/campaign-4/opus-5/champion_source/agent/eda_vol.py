"""EDA: what structure is there in the per-step return scale?

Q1  How heavy are the tails of 5-min / 1-min returns (kurtosis, and the ratio of the
    empirical-quantile scale to the standard deviation)?
Q2  How large is the intraday / day-of-week seasonality of |r|, per asset?
Q3  How persistent is volatility, and which trailing window best predicts the next 24 h?
Q4  Variance ratios across aggregation scales (microstructure noise / autocorrelation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ALL_ASSETS_24H, load_minutes  # noqa: E402

pd.set_option("display.width", 200)


def step_returns(asset: str, step_min: int) -> pd.Series:
    """Log returns of the close sampled every `step_min` minutes, NaN-aware.

    Returns spanning a NaN close are dropped (never interpolated).
    """
    c = load_minutes(asset)["close"]
    c = c.iloc[:: step_min] if step_min > 1 else c
    r = np.log(c).diff()
    return r.dropna()


def tail_stats():
    rows = []
    for a in ALL_ASSETS_24H:
        for step in (1, 5):
            r = step_returns(a, step).to_numpy()
            r = r[np.abs(r) < np.nanquantile(np.abs(r), 0.99999) * 10]
            s = r.std()
            rows.append(
                {
                    "asset": a,
                    "step": step,
                    "n": len(r),
                    "sd_bp": s * 1e4,
                    "mad/sd": np.mean(np.abs(r)) / s / np.sqrt(2 / np.pi),
                    "kurt": pd.Series(r).kurt(),
                    "q99/sd": np.quantile(np.abs(r), 0.99) / s / 2.576,
                    "q999/sd": np.quantile(np.abs(r), 0.999) / s / 3.291,
                }
            )
    return pd.DataFrame(rows)


def seasonality():
    """Ratio max/min of mean |r| across the 24 UTC hours, and weekday vs weekend."""
    rows = []
    for a in ALL_ASSETS_24H:
        r = step_returns(a, 5)
        ar = r.abs()
        byhour = ar.groupby(r.index.hour).mean()
        bydow = ar.groupby(r.index.dayofweek).mean()
        wk = bydow.loc[0:4].mean()
        we = bydow.loc[5:6].mean()
        rows.append(
            {
                "asset": a,
                "hour_max/min": byhour.max() / byhour.min(),
                "peak_hour": int(byhour.idxmax()),
                "trough_hour": int(byhour.idxmin()),
                "wkday/wkend": wk / we,
            }
        )
    return pd.DataFrame(rows)


def persistence(asset: str):
    """Regress future 24h realized vol on trailing realized vols of several lengths.

    Target and features are log realized vol of 5-min returns; the R^2 tells us how much of the
    vol level is forecastable, and the coefficients give a HAR-style blend.
    """
    r = step_returns(asset, 5)
    rv = (r**2).rolling(288, min_periods=200).mean()  # 1-day realized var, 5-min returns
    fut = (r**2).rolling(288, min_periods=200).mean().shift(-288)
    feats = {}
    for name, w in [("1h", 12), ("4h", 48), ("1d", 288), ("3d", 864), ("7d", 2016)]:
        feats[name] = np.log((r**2).rolling(w, min_periods=max(6, w // 3)).mean())
    X = pd.DataFrame(feats)
    y = np.log(fut)
    df = pd.concat([X, y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    out = {}
    for name in X.columns:
        out[f"r2_{name}"] = np.corrcoef(df[name], df["y"])[0, 1] ** 2
    A = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in X.columns])
    beta, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    pred = A @ beta
    out["r2_multi"] = 1 - np.var(df["y"] - pred) / np.var(df["y"])
    out.update({f"b_{c}": b for c, b in zip(["const"] + list(X.columns), beta)})
    out["asset"] = asset
    out["n"] = len(df)
    return out


def variance_ratios():
    """var(r_k)/(k·var(r_1)) for k-minute returns: <1 means negative autocorrelation
    (microstructure noise), >1 means trending / vol clustering at that scale."""
    rows = []
    for a in ALL_ASSETS_24H:
        base = step_returns(a, 1).var()
        row = {"asset": a, "sd_1m_bp": np.sqrt(base) * 1e4}
        for k in (2, 5, 15, 60, 300):
            row[f"vr_{k}"] = step_returns(a, k).var() / (k * base)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "tails"):
        print("\n=== TAILS ===")
        print(tail_stats().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if what in ("all", "seas"):
        print("\n=== SEASONALITY (|5-min return|) ===")
        print(seasonality().to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if what in ("all", "vr"):
        print("\n=== VARIANCE RATIOS ===")
        print(variance_ratios().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if what in ("all", "pers"):
        print("\n=== VOL PERSISTENCE (log RV, 24h ahead) ===")
        print(
            pd.DataFrame([persistence(a) for a in ALL_ASSETS_24H]).to_string(
                index=False, float_format=lambda x: f"{x:.3f}"
            )
        )
