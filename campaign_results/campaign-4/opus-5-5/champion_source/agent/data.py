"""Lenient minute-price loader over the frozen snapshot (NaN kept, never interpolated)."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
ROOT = WORKSPACE / "market_data" / "prices"
OHLCV = ["open", "high", "low", "close", "volume", "trade_count"]


@lru_cache(maxsize=32)
def load_asset(asset: str) -> pd.DataFrame:
    files = sorted((ROOT / asset / "1m").glob("date=*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["timestamp", *OHLCV]) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()
