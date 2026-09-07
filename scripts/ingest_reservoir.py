"""Ingest Hyperliquid minute closes from the Hydromancer Reservoir 1-second archive.

`synth_lib.preparation.market_data` fills the store from the venues themselves, which works for the
Binance-routed assets but stops ~3.5 days back for every Hyperliquid-routed one: that is all the
candle endpoint serves. A campaign whose `data_cutoff` is more than a few days old therefore has no
Hyperliquid history at all from the venue, and `com-equ-24h` — a third of the Score — is unfittable.

Reservoir publishes Hyperliquid history as 1-second candles, partitioned by dex and day. This script
reads them, resamples to 1-minute OHLCV, and writes the partitions `MinutePriceStore` expects,
widened with `open/high/low/volume/trade_count`. `close` keeps the venue's own definition (the last
1-second close in the minute), so the scored series is unchanged by the widening.

The archive records one close, one volume and one trade count per second, so `open`/`high`/`low` are
`first`/`max`/`min` over the 1-second closes inside each minute and `volume`/`trade_count` sum.

    uv sync --extra reservoir
    AWS_PROFILE=<profile> uv run python scripts/ingest_reservoir.py --start 2025-08-01 --end 2026-08-29

Access: the bucket is requester-pays, so reads are billed to your own AWS account and every call
carries `RequestPayer=requester`. Credentials must be in the commercial partition.

Cost: one day file serves every coin on that dex, so the reads are grouped by (dex, day) rather than
by asset — eight of the nine assets share the `xyz` dex, which is the difference between 9 reads a
day and 2. Column projection does the rest.
"""

from __future__ import annotations

import argparse
import io
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

import boto3
import botocore.config
import botocore.exceptions
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from synth_lib.preparation.config import HYPERLIQUID_SYMBOLS, MINUTES_PER_DAY
from synth_lib.preparation.minute_price_store import MinutePriceStore

UTC = timezone.utc

RESERVOIR_BUCKET = "hydromancer-reservoir"
RESERVOIR_REGION = "ap-northeast-1"
SOURCE_NAME = "hydromancer-reservoir"

# Projection is what keeps a requester-pays read small. `close` is the only column the archive is
# guaranteed to carry; the rest are taken when present and derived when not.
CANDLE_COLUMNS = ["coin", "timestamp", "close", "volume", "trade_count"]

# The widened partition. Readers select by name (`generate_predictions` takes timestamp+close only),
# so the extra columns reach agents without touching scoring.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count"]

# First day each market traded on the archive. Reads before these return nothing because the market
# did not exist — a fact about the market, not a gap. `@107` is the exception: 2025-08-01 is where
# the archive begins rather than where HYPE did.
FIRST_TRADE: dict[str, date] = {
    "HYPE": date(2025, 8, 1),
    "NVDAX": date(2025, 11, 12),
    "TSLAX": date(2025, 11, 13),
    "GOOGLX": date(2025, 11, 18),
    "AAPLX": date(2025, 11, 21),
    "XAU": date(2025, 12, 22),
    "WTIOIL": date(2026, 1, 6),
    "SP500": date(2026, 3, 18),
    "SPCX": date(2026, 5, 17),
}


def dex_for_coin(coin: str) -> str:
    """Reservoir `by_dex/` partition for a coin id: `xyz:SP500` -> `xyz`, `@107` -> `hyperliquid`."""
    return coin.split(":", 1)[0] if ":" in coin else "hyperliquid"


def candles_key(dex: str, day: date) -> str:
    return f"by_dex/{dex}/candles/1s/date={day.isoformat()}/candles.parquet"


class S3RangeFile(io.RawIOBase):
    """Seekable read-only file over an S3 object, via ranged GETs.

    `pyarrow.fs.S3FileSystem` exposes no `request_payer` option, so a requester-pays bucket cannot be
    read through it at all; this wrapper is what lets parquet do column projection over one.
    """

    def __init__(self, client: Any, bucket: str, key: str, size: int) -> None:
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, min(self._size, base + offset))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""
        end = self._pos + size - 1  # HTTP ranges are inclusive on both ends
        body = self._client.get_object(
            Bucket=self._bucket,
            Key=self._key,
            Range=f"bytes={self._pos}-{end}",
            RequestPayer="requester",
        )["Body"].read()
        self._pos += len(body)
        self.bytes_fetched += len(body)
        return body

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def reservoir_client() -> Any:
    config = botocore.config.Config(
        max_pool_connections=32,
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client("s3", region_name=RESERVOIR_REGION, config=config)


def fetch_day_minute_ohlcv(client: Any, dex: str, day: date, coins: Sequence[str]) -> dict[str, pd.DataFrame]:
    """1-minute OHLCV for `coins` on `day`, resampled from the 1-second candles.

    `close` is the last 1s close in the minute, which is by definition the venue's own 1-minute
    candle close. open/high/low are first/max/min over those same closes; volume and trade_count
    sum. Returns {} when the partition does not exist; coins absent that day are missing from the
    result rather than present and empty.
    """
    key = candles_key(dex, day)
    try:
        size = client.head_object(Bucket=RESERVOIR_BUCKET, Key=key, RequestPayer="requester")["ContentLength"]
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"].get("Code") in ("404", "NoSuchKey"):
            return {}
        raise

    source = S3RangeFile(client, RESERVOIR_BUCKET, key, size)
    table = pq.read_table(source, columns=CANDLE_COLUMNS)
    # Prices and volumes are decimal128 upstream. Route to float64 via string: Arrow's decimal ->
    # float64 is not correctly rounded (7413.9 lands on 7413.900000000001), which would leave an
    # archive-derived minute one ULP from the same price fetched as JSON from the venue.
    for name in ("close", "volume"):
        table = table.set_column(
            table.schema.get_field_index(name), name, table[name].cast(pa.string())
        )
    frame = table.to_pandas()
    frame = frame.loc[frame["coin"].isin(list(coins))]
    if frame.empty:
        return {}
    frame["close"] = frame["close"].astype("float64")
    frame["volume"] = frame["volume"].astype("float64")
    # Sort before bucketing so "first"/"last in the minute" are well defined.
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp", kind="stable")
    frame["minute"] = frame["timestamp"].dt.floor("min")

    out: dict[str, pd.DataFrame] = {}
    for coin, group in frame.groupby("coin", sort=True):
        per_minute = group.groupby("minute").agg(
            open=("close", "first"),
            high=("close", "max"),
            low=("close", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_count=("trade_count", "sum"),
        )
        per_minute.index.name = "timestamp"
        out[str(coin)] = per_minute
    print(
        f"  {dex} {day.isoformat()}: {source.bytes_fetched / 1e6:.2f} of {size / 1e6:.2f} MB, "
        f"coins={sorted(out)}",
        flush=True,
    )
    return out


def write_day(store: MinutePriceStore, day: date, minutes: pd.DataFrame) -> None:
    """Persist one asset-day: the store's partition contract, widened with OHLCV."""
    store.ensure_root()
    day_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    grid = pd.date_range(day_start, periods=MINUTES_PER_DAY, freq="1min", tz="UTC")
    # float NaN, not pd.NA: a scalar pd.NA makes the column object dtype, which reads back as None
    # and breaks float()/np.isfinite() consumers. NaN everywhere, never a 0 volume: a minute with no
    # rows is one we have no record of, which is not the same claim as "nothing traded".
    frame = minutes.reindex(grid).astype("float64")
    frame.index.name = "timestamp"
    frame = frame[OHLCV_COLUMNS]
    frame["source"] = SOURCE_NAME
    frame["ingested_at"] = datetime.now(tz=UTC).replace(microsecond=0)
    frame["is_final"] = day < datetime.now(tz=UTC).date()
    frame = frame.reset_index()
    frame.to_parquet(store.day_path(day), index=False)


def days_between(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def ingest(assets: Sequence[str], start: date, end: date, force_refresh: bool, dry_run: bool) -> None:
    client = reservoir_client()
    coin_of = {a: HYPERLIQUID_SYMBOLS[a] for a in assets}
    asset_of = {coin: asset for asset, coin in coin_of.items()}
    # One store per asset: only its path helpers are used here, never its venue client.
    stores = {a: MinutePriceStore(a) for a in assets}
    written = {a: 0 for a in assets}

    for day in days_between(start, end):
        # One read per (dex, day) serves every coin on it — 2 reads a day, not 9.
        wanted: dict[str, list[str]] = {}
        for asset, coin in coin_of.items():
            if day < FIRST_TRADE[asset]:
                continue  # the market did not exist yet
            if not force_refresh and stores[asset].day_path(day).exists():
                continue
            wanted.setdefault(dex_for_coin(coin), []).append(coin)
        if not wanted:
            continue
        if dry_run:
            targets = sorted(asset_of[c] for coins in wanted.values() for c in coins)
            print(f"  {day.isoformat()}: would read {sorted(wanted)} for {targets}")
            continue
        for dex, coins in wanted.items():
            for coin, minutes in fetch_day_minute_ohlcv(client, dex, day, coins).items():
                # Never persist an empty day: an all-NaN partition is indistinguishable from a real
                # zero-trade day to every downstream reader, and `ingest_day` will skip it forever.
                if minutes.empty:
                    continue
                asset = asset_of[coin]
                write_day(stores[asset], day, minutes)
                written[asset] += int(minutes["close"].notna().sum())

    if dry_run:
        return
    for asset, minutes in sorted(written.items()):
        print(f"{asset:7} {minutes:>9,} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=sorted(FIRST_TRADE),
        metavar="ASSET",
        help="default: every Hyperliquid-routed competition asset",
    )
    parser.add_argument("--force-refresh", action="store_true", help="re-read days already on disk")
    parser.add_argument("--dry-run", action="store_true", help="report what would be read, fetch nothing")
    args = parser.parse_args()

    unknown = [a for a in args.assets if a not in FIRST_TRADE]
    if unknown:
        raise SystemExit(f"not Hyperliquid-routed, so Reservoir has no candles for: {unknown}")
    if args.start > args.end:
        raise SystemExit("--start must be on or before --end")

    print(f"{args.start} -> {args.end}, assets={args.assets}, force_refresh={args.force_refresh}")
    ingest(args.assets, args.start, args.end, args.force_refresh, args.dry_run)


if __name__ == "__main__":
    main()
