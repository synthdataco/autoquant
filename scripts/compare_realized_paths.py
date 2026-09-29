"""Does the backtester score against the same realized path the validator did?

The backtester slices each prompt's realized path out of the local minute store and only patches
NaN holes from the validator's own arrays (`_fill_gaps_from_realized_paths`). Wherever the store
is complete the two are never compared — so a mid-minute close, a truncated fetch or a venue gap
filled by the wrong bar would move the truth a champion is measured against, silently.

The validator's array is `/validation/realized-path`, which is byte-identical to the
`real_prices` it stored for the prompt. Prompt start_times come from the offline bundle, so this
needs no database.

    uv run python scripts/compare_realized_paths.py --asset HYPE --time-length 3600 \
        --bundle offline_data/verdict_20260926_19d --days 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synth_lib.backtester.config import competition_for, slug_for
from synth_lib.preparation.validator_api import get_realized_path


def store_slice(root: Path, asset: str, start: pd.Timestamp, time_length: int, increment: int) -> np.ndarray:
    """The slice the backtester would score against, on the store's own bar labels."""
    index = pd.date_range(start, periods=time_length // increment + 1, freq=pd.Timedelta(seconds=increment))
    frames = []
    day = start.date()
    while day <= index[-1].date():
        path = root / "prices" / asset / "1m" / f"date={day.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["timestamp", "close"]))
        day += pd.Timedelta(days=1).to_pytimedelta()
    if not frames:
        return np.array([])
    prices = pd.concat(frames, ignore_index=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    return prices.set_index("timestamp").sort_index()["close"].reindex(index).to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--time-length", type=int, default=3600)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("market_data"))
    ap.add_argument("--days", type=float, default=2.0, help="how much of the bundle's tail to check")
    args = ap.parse_args()

    comp = competition_for(args.asset, args.time_length)
    scores = args.bundle / f"miner_scores_{args.asset}_{slug_for(comp)}.parquet"
    if not scores.exists():
        raise SystemExit(f"no {scores}")
    scored = (
        pd.to_datetime(pd.read_parquet(scores, columns=["scored_time"])["scored_time"], utc=True)
        .drop_duplicates()
        .sort_values()
    )
    starts = scored - pd.Timedelta(seconds=args.time_length)
    starts = starts[starts >= starts.max() - pd.Timedelta(days=args.days)]

    identical = differing = unusable = missing = 0
    worst: list[tuple[float, pd.Timestamp, int]] = []
    for start in starts:
        truth = get_realized_path(args.asset, start.to_pydatetime(), args.time_length, comp.time_increment)
        if truth is None:
            missing += 1
            continue
        theirs = truth.to_numpy(dtype=float)
        mine = store_slice(args.data_root, args.asset, start, args.time_length, comp.time_increment)
        if mine.size != theirs.size or not np.isfinite(mine).all():
            unusable += 1
            continue
        if np.array_equal(theirs, mine):
            identical += 1
        else:
            differing += 1
            gap = np.abs(theirs - mine)
            rel = np.nanmax(gap / np.where(theirs == 0, np.nan, np.abs(theirs)))
            worst.append((float(rel), start, int(np.count_nonzero(gap))))

    total = identical + differing + unusable + missing
    print(f"{args.asset} tl={args.time_length}  {total} prompts over the bundle's last {args.days} days")
    print(f"  identical to the validator's array : {identical}")
    print(f"  DIFFERING                          : {differing}")
    print(f"  store NaN or short (backtest falls back to the validator) : {unusable}")
    print(f"  no validator array available       : {missing}")
    for rel, when, n in sorted(worst, reverse=True)[:5]:
        print(f"    {when}  {n} points differ, worst {rel*1e4:.1f} bp")


if __name__ == "__main__":
    main()
