"""Fast local evaluation harness.

Purpose: iterate on the model without paying the Synth-API round trips that
`synth_lib.backtester.orchestration.backtest` needs for the *field* (rank). The CRPS itself is
computed with the very same function the validator and the official backtester call
(`calculate_total_score_for_miner`, imported through `synth_lib.backtester.scoring`), on realized
paths assembled exactly the way `backtest()` assembles them:

  * `_slice_real_prices` on the local minute store (NaN tolerated, never interpolated),
  * `_fill_gaps_from_realized_paths` from the validator's cached realized paths where the
    snapshot has holes (offline: `market_data/realized/...`).

Official, publishable numbers still go through `agent/predict.py`. This file exists so that a
sweep of 40 variants costs minutes instead of hours; it is checked against predict.py in the
journal.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
os.chdir(WORKSPACE)  # synth-lib resolves the price store relative to the cwd

from synth.validator.competition_config import (  # noqa: E402
    COM_EQU_24H,
    CRYPTO_1H,
    CRYPTO_24H,
    CompetitionConfig,
)
from synth_lib.backtester.preparation import (  # noqa: E402
    _fill_gaps_from_realized_paths,
    _slice_real_prices,
)
from synth_lib.backtester.scoring import calculate_total_score_for_miner  # noqa: E402
from synth_lib.benchmark.generate_predictions import context_end  # noqa: E402
from synth_lib.preparation.config import OHLCV_COLUMNS, STORE_SUBDIR  # noqa: E402

_OFFLINE_ENV = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"
CONTEXT_MINUTES = 7 * 24 * 60
CACHE = WORKSPACE / "miner_outputs" / "harness_cache"

COMPETITIONS = {"crypto-24h": CRYPTO_24H, "com-equ-24h": COM_EQU_24H, "crypto-1h": CRYPTO_1H}
# BTC..HYPE appear in BOTH crypto-24h and crypto-1h, so the competition is keyed on
# (asset, time_length) — never on the asset alone.
ASSET_COMPETITION_24H = {a: "crypto-24h" for a in CRYPTO_24H.asset_list} | {
    a: "com-equ-24h" for a in COM_EQU_24H.asset_list
}

# SPYX is dead (feed stopped mid-July 2026); the validator no longer prompts it.
ALL_ASSETS_24H = CRYPTO_24H.asset_list + COM_EQU_24H.asset_list


def store_root(asset: str) -> Path:
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


_MINUTE_CACHE: dict[str, pd.DataFrame] = {}


def load_minutes(asset: str, start=None, end=None) -> pd.DataFrame:
    """Whole-history minute OHLCV for one asset, lenient about gaps.

    synth-lib's strict `load_range` raises on any NaN or missing day; that is right for scoring
    and wrong for exploration, so we read the partitions directly (the sanctioned lenient
    pattern). Missing days simply do not appear; NaN stays NaN.
    """
    if asset not in _MINUTE_CACHE:
        root = store_root(asset)
        files = sorted(root.glob("date=*.parquet"))
        if not files:
            raise FileNotFoundError(f"no partitions for {asset} under {root}")
        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
        df = df[list(OHLCV_COLUMNS)].astype(float)
        # Reindex onto a gapless minute grid, matching what backtest()'s
        # download_price_data(...).resample("1min").last() hands the slicer. A missing day
        # would otherwise make .loc[a:b:5] step over the wrong rows.
        grid = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
        _MINUTE_CACHE[asset] = df.reindex(grid)
    df = _MINUTE_CACHE[asset]
    if start is not None or end is not None:
        df = df.loc[start:end]
    return df


@dataclass
class PromptSet:
    """A frozen set of (context, realized path) pairs — the unit every experiment is scored on."""

    asset: str
    time_length: int
    time_increment: int
    starts: list[pd.Timestamp]
    reals: np.ndarray  # (n_prompts, steps), NaN where unobserved
    contexts: list[pd.DataFrame]

    @property
    def competition(self) -> CompetitionConfig:
        if self.time_length != 86_400:
            return CRYPTO_1H
        return COMPETITIONS[ASSET_COMPETITION_24H[self.asset]]


def _cache_key(asset, time_length, time_increment, start, end, cadence) -> Path:
    name = f"{asset}_{time_length}_{time_increment}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_{cadence}.pkl"
    return CACHE / name


def build_prompts(
    asset: str,
    start,
    end,
    time_length: int = 86_400,
    time_increment: int = 300,
    cadence_minutes: int = 60,
    use_cache: bool = True,
) -> PromptSet:
    """Prompts on a fixed grid, with the realized path each one is scored against."""
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else pd.Timestamp(start)
    end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else pd.Timestamp(end)
    path = _cache_key(asset, time_length, time_increment, start, end, cadence_minutes)
    if use_cache and path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)

    prices = load_minutes(asset)
    close_only = prices[["close"]]
    grid = pd.date_range(start, end, freq=f"{cadence_minutes}min", tz="UTC")
    grid = [t for t in grid if t + timedelta(seconds=time_length) <= prices.index[-1]]

    steps = time_length // time_increment + 1
    prompts = []
    contexts = []
    for t in grid:
        ctx = prices.loc[: context_end(t, time_length)][-CONTEXT_MINUTES:]
        if len(ctx) < 2 * 24 * 60 or ctx["close"].notna().sum() < 200:
            continue
        try:
            # close-only, exactly like backtest(): _slice_real_prices takes .iloc[:, 0]
            real = _slice_real_prices(close_only, t, time_length, time_increment)
        except ValueError:
            continue
        prompts.append(
            {
                "file_path": "x",
                "start_time": t,
                "asset": asset,
                "time_length": time_length,
                "time_increment": time_increment,
                "real_prices": real,
            }
        )
        contexts.append(ctx)

    # _fill_gaps_from_realized_paths() calls prefetch_realized_paths() unless synth-lib thinks it
    # is offline — and our grid start times are NOT the validator's irregular prompt times, so
    # every one of them looks uncached and the prefetch tries to pull it from the API one request
    # at a time (0.6 s apart) before failing to write to the read-only mount. Declaring offline
    # mode for the duration keeps it to `RealizedPathStore.get`, which snaps to the nearest
    # cached prompt within 30 min — exactly what backtest() does with its own tolerance.
    prev = os.environ.get(_OFFLINE_ENV)
    os.environ[_OFFLINE_ENV] = str(WORKSPACE / "market_data")
    try:
        _fill_gaps_from_realized_paths(prompts, f"  [{asset}]")
    finally:
        if prev is None:
            os.environ.pop(_OFFLINE_ENV, None)
        else:
            os.environ[_OFFLINE_ENV] = prev
    reals = np.array([p["real_prices"] for p in prompts], dtype=float)
    ps = PromptSet(
        asset=asset,
        time_length=time_length,
        time_increment=time_increment,
        starts=[p["start_time"] for p in prompts],
        reals=reals,
        contexts=contexts,
    )
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(ps, fh)
    return ps


def score_paths(paths: np.ndarray, real: np.ndarray, competition: CompetitionConfig) -> float:
    """Validator CRPS for one prompt. `paths` is (num_simulations, steps), float32 like the
    validator stores."""
    total, _ = calculate_total_score_for_miner(
        np.asarray(paths, dtype=float),
        np.asarray(real, dtype=float),
        competition.time_increment,
        competition.scoring_intervals,
        competition.vol_scoring_blocks,
    )
    return float(total)


def coverage(ps: PromptSet) -> float:
    return float(np.isfinite(ps.reals).mean())
