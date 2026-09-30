"""Tests for the Reservoir ingester: dex routing, the 1s->1m resample, and the partition contract."""

from __future__ import annotations

import importlib.util
import io
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from synth_lib.benchmark.generate_predictions import load_minute_prices
from synth_lib.preparation.minute_price_store import MinutePriceStore

_SPEC = importlib.util.spec_from_file_location(
    "ingest_reservoir", Path(__file__).resolve().parents[1] / "scripts" / "ingest_reservoir.py"
)
assert _SPEC and _SPEC.loader
ingest_reservoir = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_reservoir)

UTC = timezone.utc


def day_file(rows: list[tuple[str, str, str]]) -> bytes:
    """A day partition with Reservoir's real types: decimal128 close/volume, ms timestamps.

    rows: (coin, "YYYY-MM-DD HH:MM:SS", close_str) — volume is 1.0 and trade_count 1 per row, so a
    minute's summed volume equals the number of 1-second rows it contains.
    """
    table = pa.table(
        {
            "coin": pa.array([r[0] for r in rows], pa.string()),
            "timestamp": pa.array(
                [datetime.fromisoformat(r[1]).replace(tzinfo=UTC) for r in rows],
                pa.timestamp("ms", tz="UTC"),
            ),
            "close": pa.array([Decimal(r[2]) for r in rows], pa.decimal128(20, 10)),
            "volume": pa.array([Decimal("1.0") for _ in rows], pa.decimal128(20, 10)),
            "trade_count": pa.array([1 for _ in rows], pa.uint32()),
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


class FakeS3:
    """Serves one object; records how many times it was read."""

    def __init__(self, body: bytes | None):
        self._body = body
        self.head_calls = 0
        self.get_calls = 0

    def head_object(self, **kwargs):
        self.head_calls += 1
        assert kwargs["RequestPayer"] == "requester"
        if self._body is None:
            raise ingest_reservoir.botocore.exceptions.ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {"ContentLength": len(self._body)}

    def get_object(self, **kwargs):
        self.get_calls += 1
        assert kwargs["RequestPayer"] == "requester"
        start, end = (int(x) for x in kwargs["Range"].removeprefix("bytes=").split("-"))
        return {"Body": io.BytesIO(self._body[start : end + 1])}


def test_dex_for_coin_routes_hip3_and_native():
    assert ingest_reservoir.dex_for_coin("xyz:SP500") == "xyz"
    assert ingest_reservoir.dex_for_coin("xyz:CL") == "xyz"
    assert ingest_reservoir.dex_for_coin("@107") == "hyperliquid"
    assert ingest_reservoir.dex_for_coin("BTC") == "hyperliquid"


def test_candles_key():
    assert (
        ingest_reservoir.candles_key("xyz", date(2026, 3, 18))
        == "by_dex/xyz/candles/1s/date=2026-03-18/candles.parquet"
    )


def test_minute_close_is_the_last_second_in_the_minute():
    client = FakeS3(
        day_file(
            [
                ("xyz:SP500", "2026-03-18 00:00:01", "7413.9"),
                ("xyz:SP500", "2026-03-18 00:00:59", "7414.5"),
                ("xyz:SP500", "2026-03-18 00:01:30", "7415.0"),
                ("xyz:CL", "2026-03-18 00:00:10", "61.25"),
            ]
        )
    )
    out = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500"])

    # Coin filtering keeps only what was asked for — xyz:CL shared the file but was not requested.
    assert list(out) == ["xyz:SP500"]
    frame = out["xyz:SP500"]
    assert list(frame["close"]) == [7414.5, 7415.0]
    assert list(frame.index) == [
        pd.Timestamp("2026-03-18 00:00", tz="UTC"),
        pd.Timestamp("2026-03-18 00:01", tz="UTC"),
    ]


def test_ohlc_is_derived_from_the_second_closes():
    """open/high/low come from first/max/min of the 1s closes; volume and trade_count sum."""
    client = FakeS3(
        day_file(
            [
                ("xyz:SP500", "2026-03-18 00:00:01", "7413.9"),
                ("xyz:SP500", "2026-03-18 00:00:20", "7420.0"),
                ("xyz:SP500", "2026-03-18 00:00:40", "7410.0"),
                ("xyz:SP500", "2026-03-18 00:00:59", "7414.5"),
            ]
        )
    )
    row = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500"])["xyz:SP500"].iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (7413.9, 7420.0, 7410.0, 7414.5)
    assert row["volume"] == 4.0 and row["trade_count"] == 4


def test_decimal_close_survives_the_float_cast_exactly():
    """decimal128 -> float64 is not correctly rounded; the string hop keeps 7413.9 at 7413.9."""
    client = FakeS3(day_file([("xyz:SP500", "2026-03-18 00:00:01", "7413.9")]))
    out = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500"])
    assert out["xyz:SP500"]["close"].iloc[0] == 7413.9


def test_one_read_serves_every_coin_on_the_dex():
    client = FakeS3(
        day_file(
            [
                ("xyz:SP500", "2026-03-18 00:00:01", "7413.9"),
                ("xyz:CL", "2026-03-18 00:00:02", "61.25"),
                ("xyz:GOLD", "2026-03-18 00:00:03", "4210.0"),
            ]
        )
    )
    out = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500", "xyz:CL", "xyz:GOLD"])
    assert sorted(out) == ["xyz:CL", "xyz:GOLD", "xyz:SP500"]
    assert client.head_calls == 1


def test_missing_partition_returns_empty():
    assert ingest_reservoir.fetch_day_minute_ohlcv(FakeS3(None), "xyz", date(2025, 10, 1), ["xyz:SP500"]) == {}


def test_non_404_propagates():
    class Denied(FakeS3):
        def head_object(self, **kwargs):
            raise ingest_reservoir.botocore.exceptions.ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "HeadObject"
            )

    with pytest.raises(ingest_reservoir.botocore.exceptions.ClientError):
        ingest_reservoir.fetch_day_minute_ohlcv(Denied(b""), "xyz", date(2026, 3, 18), ["xyz:SP500"])


def test_write_day_matches_the_store_partition_contract(tmp_path: Path):
    store = MinutePriceStore("SP500", root=tmp_path)
    minutes = pd.DataFrame(
        {
            "open": [7413.9, 7415.0],
            "high": [7420.0, 7416.0],
            "low": [7410.0, 7414.0],
            "close": [7414.5, 7415.5],
            "volume": [4.0, 2.0],
            "trade_count": [4.0, 2.0],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-03-18 00:00", tz="UTC"), pd.Timestamp("2026-03-18 00:01", tz="UTC")],
            name="timestamp",
        ),
    )
    ingest_reservoir.write_day(store, date(2026, 3, 18), minutes)

    frame = pd.read_parquet(store.day_path(date(2026, 3, 18)))
    assert list(frame.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "source",
        "ingested_at",
        "is_final",
    ]
    assert len(frame) == 1440  # full minute grid, NaN where no trade
    assert frame["close"].dtype == "float64"
    assert frame["close"].notna().sum() == 2
    assert frame["close"].iloc[0] == 7414.5
    # A minute with no rows is NaN, never a 0 volume: "no record" is not "nothing traded".
    assert frame["volume"].isna().sum() == 1438
    assert bool(frame["is_final"].iloc[0]) is True


def test_scoring_reader_still_sees_the_partition(tmp_path: Path):
    """The scoring reader consumes the partition this script writes, columns and all."""
    store = MinutePriceStore("SP500", root=tmp_path / "prices" / "SP500" / "1m")
    minutes = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [3.0], "trade_count": [3.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-18 00:00", tz="UTC")], name="timestamp"),
    )
    ingest_reservoir.write_day(store, date(2026, 3, 18), minutes)
    frame = load_minute_prices(
        tmp_path, "SP500", pd.Timestamp("2026-03-18 00:00", tz="UTC"), pd.Timestamp("2026-03-18 00:05", tz="UTC")
    )
    assert frame["close"].iloc[0] == 1.5
    assert (frame.iloc[0][["open", "high", "low", "volume"]] == [1.0, 2.0, 0.5, 3.0]).all()


def test_first_trade_covers_every_hyperliquid_asset():
    """A missing entry would silently skip that asset for the whole run."""
    from synth_lib.preparation.config import HYPERLIQUID_SYMBOLS

    assert set(ingest_reservoir.FIRST_TRADE) == set(HYPERLIQUID_SYMBOLS)
