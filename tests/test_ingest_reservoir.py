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


def day_file(rows: list[tuple[str, ...]], busy: bool = True) -> bytes:
    """A day partition with Reservoir's real types: decimal128 prices/volume, ms timestamps.

    rows: (coin, "YYYY-MM-DD HH:MM:SS", close_str), whose open/high/low equal the close, or
    (coin, "YYYY-MM-DD HH:MM:SS", open_str, high_str, low_str, close_str). volume is 1.0 and
    trade_count 1 per row, so a minute's summed volume equals the number of 1-second rows it contains.
    `busy` adds a coin that trades every minute of the rows' day, as a live dex does.
    """
    if busy:
        day = datetime.fromisoformat(rows[0][1]).date()
        minutes = pd.date_range(pd.Timestamp(day), periods=1440, freq="1min")
        rows = rows + [("xyz:BUSY", f"{m:%Y-%m-%d %H:%M}:30", "1") for m in minutes]
    ohlc = [r[2:] if len(r) == 6 else (r[2],) * 4 for r in rows]
    table = pa.table(
        {
            "coin": pa.array([r[0] for r in rows], pa.string()),
            "timestamp": pa.array(
                [datetime.fromisoformat(r[1]).replace(tzinfo=UTC) for r in rows],
                pa.timestamp("ms", tz="UTC"),
            ),
            "open": pa.array([Decimal(p[0]) for p in ohlc], pa.decimal128(20, 10)),
            "high": pa.array([Decimal(p[1]) for p in ohlc], pa.decimal128(20, 10)),
            "low": pa.array([Decimal(p[2]) for p in ohlc], pa.decimal128(20, 10)),
            "close": pa.array([Decimal(p[3]) for p in ohlc], pa.decimal128(20, 10)),
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


def test_ohlc_aggregates_the_second_candles():
    """First 1s open, highest 1s high, lowest 1s low, last 1s close; volume and trade_count sum."""
    client = FakeS3(
        day_file(
            [
                ("xyz:SP500", "2026-03-18 00:00:01", "7413.0", "7414.2", "7412.8", "7413.9"),
                ("xyz:SP500", "2026-03-18 00:00:20", "7419.0", "7421.5", "7418.5", "7420.0"),
                ("xyz:SP500", "2026-03-18 00:00:40", "7411.0", "7411.0", "7408.7", "7410.0"),
                ("xyz:SP500", "2026-03-18 00:00:59", "7414.0", "7414.6", "7413.9", "7414.5"),
            ]
        )
    )
    row = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500"])["xyz:SP500"].iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (7413.0, 7421.5, 7408.7, 7414.5)
    assert row["volume"] == 4.0 and row["trade_count"] == 4


def test_decimal_prices_survive_the_float_cast_exactly():
    """decimal128 -> float64 is not correctly rounded; the string hop keeps 7413.9 at 7413.9."""
    client = FakeS3(day_file([("xyz:SP500", "2026-03-18 00:00:01", "7413.9")]))
    row = ingest_reservoir.fetch_day_minute_ohlcv(client, "xyz", date(2026, 3, 18), ["xyz:SP500"])["xyz:SP500"].iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (7413.9, 7413.9, 7413.9, 7413.9)


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
    assert len(frame) == 1440  # full minute grid
    assert frame["close"].dtype == "float64"
    assert frame["close"].iloc[0] == 7414.5
    # Every later minute is the venue's flat candle at the last close, with no volume and no trades.
    flat = frame.iloc[2:]
    assert (flat[["open", "high", "low", "close"]] == 7415.5).all().all()
    assert (flat["volume"] == 0).all() and (flat["trade_count"] == 0).all()
    assert bool(frame["is_final"].iloc[0]) is True


def _minutes(rows: dict[str, float]) -> pd.DataFrame:
    """Traded minutes whose open/high/low/close are all the given price."""
    index = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in rows], name="timestamp")
    prices = list(rows.values())
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0, "trade_count": 1.0},
        index=index,
    )


def test_minutes_before_the_first_trade_carry_the_previous_close(tmp_path: Path):
    store = MinutePriceStore("SP500", root=tmp_path)
    last = ingest_reservoir.write_day(store, date(2026, 3, 19), _minutes({"2026-03-19 00:05": 7420.0}), 7414.5)
    frame = pd.read_parquet(store.day_path(date(2026, 3, 19)))
    assert (frame["close"].iloc[:5] == 7414.5).all() and (frame["trade_count"].iloc[:5] == 0).all()
    assert frame["close"].iloc[5] == 7420.0 and last == 7420.0


def test_without_a_previous_close_the_leading_minutes_stay_nan(tmp_path: Path):
    store = MinutePriceStore("SP500", root=tmp_path)
    ingest_reservoir.write_day(store, date(2026, 3, 18), _minutes({"2026-03-18 00:05": 7420.0}))
    frame = pd.read_parquet(store.day_path(date(2026, 3, 18)))
    assert frame["close"].iloc[:5].isna().all() and frame["close"].iloc[5:].notna().all()


class FakeBucket(FakeS3):
    """Serves one object per key."""

    def __init__(self, bodies: dict[str, bytes]):
        super().__init__(None)
        self._bodies = bodies

    def head_object(self, **kwargs):
        if kwargs["Key"] not in self._bodies:
            raise ingest_reservoir.botocore.exceptions.ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self._bodies[kwargs["Key"]])}

    def get_object(self, **kwargs):
        self._body = self._bodies[kwargs["Key"]]
        return super().get_object(**kwargs)


def _run_ingest(
    tmp_path,
    monkeypatch,
    bodies: dict[date, bytes],
    start: date,
    end: date,
    force_refresh: bool = True,
    keep_going: bool = False,
) -> MinutePriceStore:
    monkeypatch.chdir(tmp_path)
    keys = {ingest_reservoir.candles_key("xyz", d): body for d, body in bodies.items()}
    monkeypatch.setattr(ingest_reservoir, "reservoir_client", lambda: FakeBucket(keys))
    ingest_reservoir.ingest(["SP500"], start, end, force_refresh=force_refresh, dry_run=False, keep_going=keep_going)
    return MinutePriceStore("SP500")


def test_the_close_carries_across_midnight_within_a_run(tmp_path: Path, monkeypatch):
    bodies = {
        date(2026, 3, 18): day_file([("xyz:SP500", "2026-03-18 23:58:10", "7414.5")]),
        date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 00:05:10", "7420.0")]),
    }
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 18), date(2026, 3, 19))
    second = pd.read_parquet(store.day_path(date(2026, 3, 19)))
    assert (second["close"].iloc[:5] == 7414.5).all()


def test_the_first_day_of_a_run_starts_from_the_stored_previous_day(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ingest_reservoir.write_day(MinutePriceStore("SP500"), date(2026, 3, 18), _minutes({"2026-03-18 23:58": 7414.5}))
    bodies = {date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 00:05:10", "7420.0")])}
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 19), date(2026, 3, 19))
    assert (pd.read_parquet(store.day_path(date(2026, 3, 19)))["close"].iloc[:5] == 7414.5).all()


def test_a_missing_day_is_not_bridged_by_the_close_before_it(tmp_path: Path, monkeypatch):
    bodies = {
        date(2026, 3, 18): day_file([("xyz:SP500", "2026-03-18 23:58:10", "7414.5")]),
        date(2026, 3, 20): day_file([("xyz:SP500", "2026-03-20 00:05:10", "7420.0")]),
    }
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 18), date(2026, 3, 20))
    assert not store.day_path(date(2026, 3, 19)).exists()
    assert pd.read_parquet(store.day_path(date(2026, 3, 20)))["close"].iloc[:5].isna().all()


def test_a_missing_day_starts_the_next_from_its_stored_partition(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ingest_reservoir.write_day(MinutePriceStore("SP500"), date(2026, 3, 19), _minutes({"2026-03-19 23:58": 7416.0}))
    bodies = {
        date(2026, 3, 18): day_file([("xyz:SP500", "2026-03-18 23:58:10", "7414.5")]),
        date(2026, 3, 20): day_file([("xyz:SP500", "2026-03-20 00:05:10", "7420.0")]),
    }
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 18), date(2026, 3, 20))
    assert (pd.read_parquet(store.day_path(date(2026, 3, 20)))["close"].iloc[:5] == 7416.0).all()


def _paused_day_file(day: str) -> bytes:
    """A day file whose dex trades nothing from 09:00 to 09:19."""
    rows = [
        ("xyz:BUSY", f"{day} {h:02d}:{m:02d}:30", "1") for h in range(24) for m in range(60) if not (h == 9 and m < 20)
    ]
    return day_file(rows, busy=False)


def test_a_long_pause_across_the_whole_dex_is_refused():
    with pytest.raises(ingest_reservoir.IncompleteDayError, match="20 minutes"):
        ingest_reservoir.fetch_day_minute_ohlcv(
            FakeS3(_paused_day_file("2026-03-18")), "xyz", date(2026, 3, 18), ["xyz:BUSY"]
        )


def test_a_refused_day_stops_the_run_by_default(tmp_path: Path, monkeypatch):
    bodies = {
        date(2026, 3, 18): _paused_day_file("2026-03-18"),
        date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 00:05:10", "7420.0")]),
    }
    with pytest.raises(ingest_reservoir.IncompleteDayError):
        _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 18), date(2026, 3, 19))
    assert not MinutePriceStore("SP500").day_path(date(2026, 3, 19)).exists()


def test_keep_going_logs_a_refused_day_and_ingests_the_next(tmp_path: Path, monkeypatch, capsys):
    bodies = {
        date(2026, 3, 18): _paused_day_file("2026-03-18"),
        date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 00:05:10", "7420.0")]),
    }
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 18), date(2026, 3, 19), keep_going=True)
    assert not store.day_path(date(2026, 3, 18)).exists()
    assert pd.read_parquet(store.day_path(date(2026, 3, 19)))["close"].iloc[5] == 7420.0
    out = capsys.readouterr().out
    assert "refused: by_dex/xyz/candles/1s/date=2026-03-18/candles.parquet" in out
    assert "refused 1 day file(s): xyz 2026-03-18" in out


def _store_partial_day(day: date, close: float, settled: bool) -> None:
    """A partition holding one traded minute at 00:05, settled or not."""
    store = MinutePriceStore("SP500")
    ingest_reservoir.write_day(store, day, _minutes({f"{day.isoformat()} 00:05": close}))
    frame = pd.read_parquet(store.day_path(day))
    frame["is_final"] = settled
    frame.to_parquet(store.day_path(day), index=False)


def test_a_settled_day_on_disk_is_skipped(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _store_partial_day(date(2026, 3, 19), 7416.0, settled=True)
    bodies = {date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 12:00:10", "7500.0")])}
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 19), date(2026, 3, 19), force_refresh=False)
    assert (pd.read_parquet(store.day_path(date(2026, 3, 19)))["close"].iloc[5:] == 7416.0).all()


def test_an_unsettled_day_on_disk_is_read_again(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _store_partial_day(date(2026, 3, 19), 7416.0, settled=False)
    bodies = {date(2026, 3, 19): day_file([("xyz:SP500", "2026-03-19 12:00:10", "7500.0")])}
    store = _run_ingest(tmp_path, monkeypatch, bodies, date(2026, 3, 19), date(2026, 3, 19), force_refresh=False)
    frame = pd.read_parquet(store.day_path(date(2026, 3, 19)))
    assert (frame["close"].iloc[720:] == 7500.0).all()
    # With no stored previous day, the stored closes before the archive's first trade stay.
    assert frame["close"].iloc[:5].isna().all() and (frame["close"].iloc[5:720] == 7416.0).all()
    assert frame["is_final"].all()


def test_a_failed_write_leaves_the_stored_partition_whole(tmp_path: Path, monkeypatch):
    store = MinutePriceStore("SP500", root=tmp_path)
    ingest_reservoir.write_day(store, date(2026, 3, 19), _minutes({"2026-03-19 00:05": 7416.0}))

    def killed(self, path, **kwargs):
        Path(path).write_bytes(b"PAR1")
        raise OSError("killed mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", killed)
    with pytest.raises(OSError):
        ingest_reservoir.write_day(store, date(2026, 3, 19), _minutes({"2026-03-19 00:05": 7500.0}))
    assert pd.read_parquet(store.day_path(date(2026, 3, 19)))["close"].iloc[5] == 7416.0


def test_a_short_pause_across_the_whole_dex_is_a_quiet_market():
    rows = [
        ("xyz:BUSY", f"2026-03-18 {h:02d}:{m:02d}:30", "1")
        for h in range(24)
        for m in range(60)
        if not (h == 9 and m < 2)
    ]
    out = ingest_reservoir.fetch_day_minute_ohlcv(
        FakeS3(day_file(rows, busy=False)), "xyz", date(2026, 3, 18), ["xyz:BUSY"]
    )
    assert len(out["xyz:BUSY"]) == 1438


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
