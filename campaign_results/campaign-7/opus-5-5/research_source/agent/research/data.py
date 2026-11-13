"""Research helpers: load an asset's full minute history with the sanctioned lenient loader."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from synth_lib.benchmark.generate_predictions import load_minute_prices

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE / "market_data"
ASSETS_CRYPTO = ["BTC", "ETH", "SOL", "XRP", "HYPE"]
ASSETS_EQU = ["XAU", "NVDAX", "GOOGLX", "TSLAX", "AAPLX", "WTIOIL", "SPCX", "SP500"]
ALL_ASSETS = ASSETS_CRYPTO + ASSETS_EQU


def first_day(asset: str) -> pd.Timestamp:
    root = DATA_ROOT / "prices" / asset / "1m"
    return pd.Timestamp(min(p.stem.removeprefix("date=") for p in root.glob("date=*.parquet")), tz="UTC")


def load_all(asset: str, start: str | None = None, end: str = "2026-11-08 23:59") -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC") if start else first_day(asset)
    return load_minute_prices(DATA_ROOT, asset, s, pd.Timestamp(end, tz="UTC"))
