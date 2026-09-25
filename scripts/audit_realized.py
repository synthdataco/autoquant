"""Audit the cached realized paths — the validator's own arrays, per prompt.

Wherever a prompt's slice of the local minute store contains NaN, the backtester substitutes the
validator's realized path for that prompt. Without a cached path it cannot, and the prompt is
scored from the holed local slice instead: NaN scoring points drop out of the CRPS sum, so a
thinner store yields a smaller sum and flatters whoever is being scored.

That fallback once ran for the Hyperliquid-routed assets only, on the reasoning that they are the
ones whose minute history the venue cannot serve. It also decided, silently, which source of truth
each asset was measured against — and the assets with the best local data were the ones with no
safety net. This script fails if any asset the verdict will score lacks paths over the window.

    uv run python scripts/audit_realized.py --cutoff 2026-09-14 --days 32

`--days` is the `--days` passed to build_offline_bundle, since that is what decides how far back
the paths were fetched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from synth_lib.benchmark.verdict.evaluate import COMPETITIONS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True, type=date.fromisoformat, metavar="YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=32, help="the bundle's --days (default: 32)")
    ap.add_argument("--realized", type=Path, default=Path("market_data/realized"))
    args = ap.parse_args()

    start = args.cutoff - timedelta(days=args.days)
    want = (args.cutoff - start).days + 1
    print(f"{args.realized}   {start} .. {args.cutoff}  ({want} days)\n")

    failed = 0
    for comp in COMPETITIONS:
        shape = f"{comp.time_length}_{comp.time_increment}"
        print(f"{comp.label}  ({shape})")
        for asset in comp.asset_list:
            day_dir = args.realized / asset / shape
            if not day_dir.is_dir():
                print(f"  ABSENT  {asset:7} no cached paths at all")
                failed += 1
                continue
            have = [
                d
                for d in (date.fromisoformat(f.stem[5:]) for f in day_dir.glob("date=*.parquet"))
                if start <= d <= args.cutoff
            ]
            if len(have) < want:
                print(f"  SHORT   {asset:7} {len(have)}/{want} days")
                failed += 1
        print()
    if failed:
        print(f"{failed} asset/competition pair(s) without paths over the window.")
        print("Build them with: build_offline_bundle --competition <slug> --days <n> --eval-end <cutoff>")
    else:
        print("every asset has realized paths over the whole window")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
