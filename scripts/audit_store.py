"""Audit the minute store over a campaign's window, by reading inside the partitions.

`build_snapshot` hard-links the partitions that exist and `manifest.json` records only
`date=*.parquet` names, so a store full of half-written days passes every structural check there
is. The agents then backtest against it, and a CRPS is a SUM over scoring points with NaN points
dropped — a thinner reference yields a smaller sum, so holes make a champion's own backtest look
better than it is. It nominates on that, and the verdict scores it correctly.

    uv run python scripts/audit_store.py --cutoff 2026-09-14
    uv run python scripts/audit_store.py --cutoff 2026-09-14 --store campaign_runs/c4/snapshot/prices

Two severities. DAMAGE is a store defect with a known repair and exits non-zero. REVIEW is a fact
about the market that only the operator can judge: every asset here trades 24/7, so a NaN minute is
a minute in which nothing traded, not a session gap. Thin assets have many; how many is too many
for a 24-hour horizon is a judgement, not a rule.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from synth_lib.preparation.config import ALL_SYMBOLS, BINANCE_SYMBOLS, MINUTES_PER_DAY


def audit_asset(asset: str, store: Path, start: date, cutoff: date) -> tuple[list[str], list[str]]:
    """Returns (damage, review) findings for one asset."""
    day_dir = store / asset / "1m"
    files = sorted(day_dir.glob("date=*.parquet")) if day_dir.is_dir() else []
    files = [f for f in files if start <= date.fromisoformat(f.stem[5:]) <= cutoff]
    if not files:
        return [f"no partitions between {start} and {cutoff}"], []

    damage: list[str] = []
    review: list[str] = []
    empty_days: list[str] = []
    truncated: list[tuple[str, int]] = []
    holed: list[tuple[str, int]] = []
    total = missing = 0

    for path in files:
        closes = pd.read_parquet(path, columns=["close"])["close"]
        nan = int(closes.isna().sum())
        total += len(closes)
        missing += nan
        if nan == 0:
            continue
        day = path.stem[5:]
        if nan == len(closes):
            empty_days.append(day)
        else:
            last = closes.last_valid_index()
            if last is not None and last < len(closes) - 1 and closes.iloc[last + 1 :].isna().all():
                truncated.append((day, int(last)))
            else:
                holed.append((day, nan))

    first, last_day = (date.fromisoformat(f.stem[5:]) for f in (files[0], files[-1]))
    gaps = (last_day - first).days + 1 - len(files)
    if gaps:
        damage.append(f"{gaps} day(s) with no file between {first} and {last_day}")
    if last_day < cutoff:
        damage.append(f"stops {(cutoff - last_day).days}d short of the cutoff")
    if empty_days:
        # A venue request for a day past its retention returns nothing and `ingest_day` persists
        # 1440 NaN rows marked final, which every later ingest then skips.
        damage.append(f"{len(empty_days)} all-NaN day(s), first {empty_days[0]} — re-ingest from the archive")
    if truncated:
        # `ingest_day` recomputes finality from the calendar rather than the partition, so a day
        # written while it was still today stays partial for good.
        damage.append(
            f"{len(truncated)} truncated day(s), first {truncated[0][0]} after minute {truncated[0][1]}"
            " — --force-refresh those dates"
        )
    if holed:
        worst = max(holed, key=lambda x: x[1])
        share = f"{100 * missing / total:.2f}%" if total else "n/a"
        note = f"{len(holed)} day(s) with untraded minutes, worst {worst[0]} ({worst[1]}/{MINUTES_PER_DAY}), {share} overall"
        # The majors are quoted continuously and deeply; a gap there is far more likely a short
        # fetch than an empty minute, so it is worth a second look even though it is not proof.
        (damage if asset in BINANCE_SYMBOLS else review).append(note)
    return damage, review


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True, type=date.fromisoformat, metavar="YYYY-MM-DD")
    ap.add_argument("--start", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--store", type=Path, default=Path("market_data/prices"))
    ap.add_argument("--days", type=int, default=None, help="audit only this many days back from the cutoff")
    args = ap.parse_args()

    start = args.start or (args.cutoff - timedelta(days=args.days) if args.days else date(2025, 8, 1))
    print(f"{args.store}   {start} .. {args.cutoff}\n")
    failed = flagged = 0
    for asset in sorted(ALL_SYMBOLS):
        damage, review = audit_asset(asset, args.store, start, args.cutoff)
        failed += bool(damage)
        flagged += bool(review)
        if damage:
            print(f"  DAMAGE  {asset:7} " + "; ".join(damage))
        for note in review:
            print(f"  review  {asset:7} {note}")
    if not failed and not flagged:
        print("  every asset complete to the cutoff")
    print(f"\n{failed} asset(s) with damage, {flagged} for review.")
    print("Damage has a repair; review is a liquidity fact only you can judge.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
