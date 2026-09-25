"""Audit the minute store over a campaign's window, by reading inside the partitions.

`build_snapshot` hard-links the partitions that exist and `manifest.json` records only
`date=*.parquet` names, so a store full of half-written days passes every structural check there
is. The agents then backtest against it, and a CRPS is a SUM over scoring points with NaN points
dropped — a thinner reference yields a smaller sum, so holes make a champion's own backtest look
better than it is. It nominates on that, and the verdict scores it correctly.

    uv run python scripts/audit_store.py --cutoff 2026-09-14
    uv run python scripts/audit_store.py --cutoff 2026-09-14 --store campaign_runs/c4/snapshot/prices

Two severities. DAMAGE is a store defect with a known repair and exits non-zero — a day the
partition itself marks unsettled, a day of pure NaN, a missing file, a store short of the
cutoff. REVIEW is a fact
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


def audit_asset(
    asset: str, store: Path, start: date, cutoff: date
) -> tuple[list[str], list[str], tuple[date, date] | None]:
    """Returns (damage, review, complete_from) for one asset.

    The third value is (first partition in the window, start of the unbroken run back from the
    cutoff). When they differ there is a hole between them; when they match, that is simply where
    this asset's history begins.
    """
    day_dir = store / asset / "1m"
    files = sorted(day_dir.glob("date=*.parquet")) if day_dir.is_dir() else []
    files = [f for f in files if start <= date.fromisoformat(f.stem[5:]) <= cutoff]
    if not files:
        return [f"no partitions between {start} and {cutoff}"], [], None

    damage: list[str] = []
    review: list[str] = []
    empty_days: list[str] = []
    partial: list[tuple[str, int]] = []
    holed: list[tuple[str, int, int]] = []
    total = missing = 0

    for path in files:
        frame = pd.read_parquet(path, columns=["close", "is_final"])
        closes = frame["close"]
        nan = int(closes.isna().sum())
        total += len(closes)
        missing += nan
        day = path.stem[5:]
        # The partition records whether the day was settled when it was written. `ingest_day`
        # recomputes that from the calendar and never reads it back, so a day written while it was
        # still running stays partial for good — and this column is the only place it says so.
        if not bool(frame["is_final"].iloc[0]):
            partial.append((day, int(closes.last_valid_index() or -1)))
            continue
        if nan == 0:
            continue
        if nan == len(closes):
            empty_days.append(day)
        else:
            last = closes.last_valid_index()
            tail = len(closes) - 1 - int(last) if last is not None else len(closes)
            holed.append((day, nan, tail))

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
    if partial:
        damage.append(
            f"{len(partial)} unsettled day(s) (is_final=False), first {partial[0][0]} after minute"
            f" {partial[0][1]} — --force-refresh those dates"
        )
    if holed:
        worst = max(holed, key=lambda x: x[1])
        longest = max(holed, key=lambda x: x[2])
        share = f"{100 * missing / total:.2f}%" if total else "n/a"
        note = (
            f"{len(holed)} day(s) with untraded minutes, worst {worst[0]} ({worst[1]}/{MINUTES_PER_DAY}),"
            f" longest run to end-of-day {longest[2]}min on {longest[0]}, {share} overall"
        )
        # The majors are quoted continuously and deeply; a gap there is far more likely a short
        # fetch than an empty minute, so it is worth a second look even though it is not proof.
        (damage if asset in BINANCE_SYMBOLS else review).append(note)
    have = {date.fromisoformat(f.stem[5:]) for f in files}
    if cutoff not in have:
        return damage, review, None
    complete_from = cutoff
    while (earlier := complete_from - timedelta(days=1)) in have:
        complete_from = earlier
    return damage, review, (min(have), complete_from)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True, type=date.fromisoformat, metavar="YYYY-MM-DD")
    ap.add_argument("--start", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--store", type=Path, default=Path("market_data/prices"))
    ap.add_argument("--days", type=int, default=None, help="audit only this many days back from the cutoff")
    ap.add_argument(
        "--require-from",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="fail unless every asset is complete back to this day",
    )
    args = ap.parse_args()

    start = args.start or (args.cutoff - timedelta(days=args.days) if args.days else date(2025, 8, 1))
    print(f"{args.store}   {start} .. {args.cutoff}\n")
    failed = flagged = 0
    complete: dict[str, date | None] = {}
    for asset in sorted(ALL_SYMBOLS):
        damage, review, complete_from = audit_asset(asset, args.store, start, args.cutoff)
        complete[asset] = complete_from
        failed += bool(damage)
        flagged += bool(review)
        if damage:
            print(f"  DAMAGE  {asset:7} " + "; ".join(damage))
        for note in review:
            print(f"  review  {asset:7} {note}")
    if not failed and not flagged:
        print("  no damage, nothing to review")

    print(f"\n{failed} asset(s) with damage, {flagged} for review.")
    print("Damage has a repair; review is a liquidity fact only you can judge.\n")

    gaps = [a for a, d in complete.items() if d is None]
    if gaps:
        print(f"  no unbroken run back from the cutoff: {', '.join(sorted(gaps))}")
        failed += len(gaps)
    else:
        runs = {a: v[1] for a, v in complete.items()}  # type: ignore[index]
        binding = max(runs, key=lambda a: runs[a])
        print(f"  all {len(runs)} assets complete from {runs[binding]}  (bound by {binding})")
        if args.require_from:
            late = {a: v for a, v in complete.items() if v[1] > args.require_from}  # type: ignore[index]
            if late:
                print(f"\n  REQUIRED from {args.require_from}, but these start later:")
                for asset, (seen, run) in sorted(late.items(), key=lambda kv: kv[1][1], reverse=True):
                    why = "history starts here" if seen == run else f"GAP — partitions exist from {seen}"
                    print(f"      {asset:7} {run}   ({why})")
                failed += len(late)
            else:
                print(f"  requirement met: every asset complete from {args.require_from}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
