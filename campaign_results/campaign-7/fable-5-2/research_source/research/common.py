"""Shared research helpers: cached full-history minute frames per asset (read-only snapshot -> scratch cache)."""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
SCRATCH = WS / "scratch"
ASSETS_CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
ASSETS_COMEQ = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
ALL_ASSETS = ASSETS_CRYPTO + ASSETS_COMEQ
COEF = {"BTC": 1.0, "ETH": 0.7064366394033871, "XAU": 1.7370922597118699, "SOL": 0.6310037175639559,
        "SP500": 3.437935601155441, "NVDAX": 1.6028217601617174, "TSLAX": 1.6068755936957768,
        "AAPLX": 2.0916380815843123, "GOOGLX": 1.6827392777257926, "XRP": 0.5658394110809131,
        "HYPE": 0.4784547133706857, "WTIOIL": 0.8475062847978935, "SPCX": 0.4329342627683478}
OHLCV = ["open", "high", "low", "close", "volume", "trade_count"]

def load_asset(asset: str, refresh: bool = False) -> pd.DataFrame:
    """Full minute history of one asset on a complete minute grid (NaN where no price)."""
    cache = SCRATCH / f"prices_{asset}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    files = sorted(glob.glob(str(WS / "market_data" / "prices" / asset / "1m" / "date=*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["timestamp", *OHLCV]) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    grid = pd.date_range(df.index[0].floor("D"), df.index[-1].floor("D") + pd.Timedelta(minutes=1439), freq="1min")
    df = df.reindex(grid)
    df.index.name = "timestamp"
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    SCRATCH.mkdir(exist_ok=True)
    df.to_parquet(cache)
    return df
