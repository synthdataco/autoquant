"""Audit the minute store over a campaign's window, by reading inside the partitions.

`build_snapshot` hard-links the partitions that exist and `manifest.json` records only
`date=*.parquet` names, so a store of half-written days passes every structural check there is. The
agents then backtest against it, and a CRPS is a SUM over scoring points with NaN points dropped —
a thinner reference yields a smaller sum, so holes make a champion's own backtest look better than
it is. It nominates on that, and the verdict scores it correctly.

    uv run python scripts/audit_store.py --cutoff 2026-09-14
    uv run python scripts/audit_store.py --cutoff 2026-09-14 --store campaign_runs/c4/snapshot/prices

Every asset prints a row whether or not it has findings: an asset missing from the output would be
indistinguishable from one that was never read.

Two severities. DAMAGE is a store defect with a known repair and exits non-zero — a day the
partition itself marks unsettled, a day of pure NaN, a missing file, a store short of the cutoff.
REVIEW is a fact about the market that only the operator can judge: every asset here trades 24/7,
so a NaN minute is a minute in which nothing traded, not a session gap. Thin assets have many; how
many is too many at a 24-hour horizon is a judgement, not a rule.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from synth_lib.preparation.config import ALL_SYMBOLS, BINANCE_SYMBOLS, MINUTES_PER_DAY


@dataclass
class Audit:
    asset: str
    days: int = 0
    first: date | None = None
    complete_from: date | None = None
    untraded_share: float = 0.0
    damage: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)

    @property
    def state(self) -> str:
        return "DAMAGE" if self.damage else ("review" if self.review else "ok")

    @property
    def has_gap(self) -> bool:
        """Partitions exist further back than the unbroken run: a hole, not a listing date."""
        return self.first is not None and self.complete_from is not None and self.first < self.complete_from


def audit_asset(asset: str, store: Path, start: date, cutoff: date) -> Audit:
    out = Audit(asset)
    day_dir = store / asset / "1m"
    files = sorted(day_dir.glob("date=*.parquet")) if day_dir.is_dir() else []
    files = [f for f in files if start <= date.fromisoformat(f.stem[5:]) <= cutoff]
    if not files:
        out.damage.append(f"no partitions between {start} and {cutoff}")
        return out

    empty_days: list[str] = []
    unsettled: list[tuple[str, int]] = []
    holed: list[tuple[str, int, int]] = []
    short: list[tuple[str, int]] = []
    total = missing = 0
    for path in files:
        frame = pd.read_parquet(path, columns=["close", "is_final"])
        closes = frame["close"]
        nan = int(closes.isna().sum())
        total += len(closes)
        missing += nan
        day = path.stem[5:]
        # A partition is a full minute grid, NaN where nothing traded. A short one is not a thin
        # day: its missing minutes are absent rather than null, so every count below would read it
        # as healthy — the shape this audit exists to reject.
        if len(closes) != MINUTES_PER_DAY:
            short.append((day, len(closes)))
        # The partition records whether the day was settled when it was written. `ingest_day`
        # recomputes that from the calendar and never reads it back, so a day written while it was
        # still running stays partial for good — and this column is the only place it says so.
        if not bool(frame["is_final"].iloc[0]):
            unsettled.append((day, int(closes.last_valid_index() or -1)))
        elif nan == len(closes):
            empty_days.append(day)
        elif nan:
            last = closes.last_valid_index()
            holed.append((day, nan, len(closes) - 1 - int(last) if last is not None else len(closes)))

    have = {date.fromisoformat(f.stem[5:]) for f in files}
    out.days = len(files)
    out.first = min(have)
    out.untraded_share = 100 * missing / total if total else 0.0
    if cutoff in have:
        run = cutoff
        while (earlier := run - timedelta(days=1)) in have:
            run = earlier
        out.complete_from = run

    if out.complete_from is None:
        out.damage.append(f"no partition for the cutoff {cutoff}")
    gaps = (max(have) - out.first).days + 1 - len(files)
    if gaps:
        out.damage.append(f"{gaps} day(s) with no file between {out.first} and {max(have)}")
    if short:
        out.damage.append(
            f"{len(short)} partition(s) short of {MINUTES_PER_DAY} rows, first {short[0][0]}"
            f" ({short[0][1]} rows) — --force-refresh those dates"
        )
    if empty_days:
        out.damage.append(f"{len(empty_days)} all-NaN day(s), first {empty_days[0]} — re-ingest from the archive")
    if unsettled:
        out.damage.append(
            f"{len(unsettled)} unsettled day(s) (is_final=False), first {unsettled[0][0]} after minute"
            f" {unsettled[0][1]} — --force-refresh those dates"
        )
    if holed:
        worst = max(holed, key=lambda x: x[1])
        longest = max(holed, key=lambda x: x[2])
        note = (
            f"{len(holed)} day(s) with untraded minutes, worst {worst[0]} ({worst[1]}/{MINUTES_PER_DAY}),"
            f" longest run to end-of-day {longest[2]}min on {longest[0]}"
        )
        # The majors are quoted continuously and deeply, so a gap there is far more likely a short
        # fetch than an empty minute — worth stopping for even though it is not proof.
        (out.damage if asset in BINANCE_SYMBOLS else out.review).append(note)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True, type=date.fromisoformat, metavar="YYYY-MM-DD")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2025, 8, 1), metavar="YYYY-MM-DD")
    ap.add_argument("--store", type=Path, default=Path("market_data/prices"))
    ap.add_argument(
        "--require-from",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="fail unless every asset has an unbroken run back to this day",
    )
    args = ap.parse_args()

    print(f"{args.store}   {args.start} .. {args.cutoff}\n")
    audits = [audit_asset(a, args.store, args.start, args.cutoff) for a in sorted(ALL_SYMBOLS)]

    print(f"{'asset':8} {'first':>11} {'complete from':>14} {'days':>6} {'untraded':>9}  state")
    for a in audits:
        run = a.complete_from.isoformat() if a.complete_from else "—"
        first = a.first.isoformat() if a.first else "—"
        print(f"{a.asset:8} {first:>11} {run:>14} {a.days:>6} {a.untraded_share:>8.2f}%  {a.state}")

    findings = [a for a in audits if a.damage or a.review]
    if findings:
        print()
        for a in findings:
            for note in a.damage:
                print(f"  DAMAGE  {a.asset:7} {note}")
            for note in a.review:
                print(f"  review  {a.asset:7} {note}")

    failed = sum(bool(a.damage) for a in audits)
    print(f"\n{failed} asset(s) with damage, {sum(bool(a.review) for a in audits)} for review.")
    print("Damage has a repair; review is a liquidity fact only you can judge.\n")

    runs = {a.asset: a.complete_from for a in audits if a.complete_from}
    if len(runs) < len(audits):
        print(f"  no unbroken run to the cutoff: {', '.join(a.asset for a in audits if not a.complete_from)}")
    else:
        binding = max(runs, key=lambda k: runs[k])
        print(f"  all {len(runs)} assets complete from {runs[binding]}  (bound by {binding})")
        if args.require_from:
            late = [a for a in audits if a.complete_from and a.complete_from > args.require_from]
            if late:
                print(f"\n  REQUIRED from {args.require_from}, but these start later:")
                for a in sorted(late, key=lambda x: x.complete_from, reverse=True):
                    why = f"GAP — partitions exist from {a.first}" if a.has_gap else "history starts here"
                    print(f"      {a.asset:7} {a.complete_from}   ({why})")
                failed += len(late)
            else:
                print(f"  requirement met: every asset complete from {args.require_from}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
