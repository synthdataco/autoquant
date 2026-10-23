# DATA.md — the shape of your market data (generated from THIS snapshot at setup)

Each day is one parquet file, `date=YYYY-MM-DD.parquet`, a full 1440-row minute grid
with columns `open, high, low, close, volume, trade_count`. `close` is the scored series.
`close` is the venue's 1-minute candle close (the last trade of that minute); the other
columns describe the same minute. For the Hyperliquid-routed assets open/high/low are
derived from 1-second closes, so a minute's range is sampled rather than continuous.
`source`/`ingested_at`/`is_final` are ingestion provenance — ignore them. Semantics:

- **NaN close** = no trade that minute (thin market or feed gap). NOT market hours:
  these venues trade 24/7. Real density varies by asset and era — see the table.
- **Missing file** = before that market's first trade. Nothing existed; nothing is hidden.
- Scoring drops NaN per prompt. Never interpolate: inventing prices in the gaps is how
  you fool your own validation.

Per-asset coverage, measured from this snapshot (nan% = share of NaN minutes among the
asset's real days; first/last 30d columns show density drift — early months of newer
listings are genuinely thin, low realized volatility there is real):

| asset | first day | last day | real days | nan% total | nan% first 30d | nan% last 30d |
|---|---|---|---|---|---|---|
| AAPLX | 2025-11-21 | 2026-10-11 | 325 | 41.1% | 60.1% | 27.0% |
| BTC | 2025-08-01 | 2026-10-11 | 437 | 0.0% | 0.0% | 0.0% |
| ETH | 2025-08-01 | 2026-10-11 | 437 | 0.0% | 0.0% | 0.0% |
| GOOGLX | 2025-11-18 | 2026-10-11 | 328 | 23.2% | 23.9% | 2.1% |
| HYPE | 2025-08-01 | 2026-10-11 | 437 | 0.1% | 0.1% | 0.0% |
| NVDAX | 2025-11-12 | 2026-10-11 | 334 | 10.3% | 14.9% | 6.3% |
| SOL | 2025-08-01 | 2026-10-11 | 437 | 0.0% | 0.0% | 0.0% |
| SP500 | 2026-03-18 | 2026-10-11 | 208 | 1.0% | 2.8% | 0.9% |
| SPCX | 2026-05-17 | 2026-10-11 | 148 | 2.9% | 4.6% | 6.3% |
| TSLAX | 2025-11-13 | 2026-10-11 | 333 | 14.8% | 26.2% | 16.5% |
| WTIOIL | 2026-01-06 | 2026-10-11 | 279 | 11.3% | 58.8% | 0.6% |
| XAU | 2025-12-22 | 2026-10-11 | 294 | 10.5% | 39.2% | 7.1% |
| XRP | 2025-08-01 | 2026-10-11 | 437 | 0.0% | 0.0% | 0.0% |

`timestamp` loads as `datetime64[us, UTC]`. An integer view of it (`.view("int64")`) counts in that unit:
derive minute indices from the dtype, never by assuming nanoseconds.

**Cached realized paths** (13 assets): **2026-06-20 → 2026-10-11**. Where a
prompt's slice of the price store contains NaN, scoring substitutes the validator's own
array for that prompt. Outside this range it cannot, and the prompt is scored on the
points that remain — which is a smaller CRPS from a thinner reference, not a better model.

Steps with no price change in those paths, the arrays CRPS is computed against: XAU (5 min) 11.3%, SOL (1 min) 9.9%, XRP (1 min) 8.0%, SPCX (5 min) 6.8%, BTC (1 min) 5.8%, AAPLX (5 min) 5.4%, NVDAX (5 min) 5.4%, SP500 (5 min) 5.2%, SOL (5 min) 4.1%, GOOGLX (5 min) 4.0%, TSLAX (5 min) 3.3%, XRP (5 min) 3.0%, ETH (1 min) 2.9%, HYPE (1 min) 1.3%, WTIOIL (5 min) 0.9%, BTC (5 min) 0.7%, ETH (5 min) 0.4%, HYPE (5 min) 0.3%.

**Scheduled events** (`calendar/events.parquet`; your champion reads it at `$SYNTH_BENCHMARK_CALENDAR`):
200 events, 2025-08-01 → 2027-01-18: every date
published by the cutoff, upcoming ones included. Columns: `time_utc`, `event`, `scope` (`all`, or the
assets it concerns, comma-separated), `detail`, `source`. Times are scheduled New York times in UTC,
so they move an hour with daylight saving: US macro data at 08:30, FOMC statements at 14:00, the EIA
petroleum report Wednesdays at 10:30 (holiday weeks differ). Earnings are anchored at the session,
08:00 before the open and 16:00 after the close; an NYSE closure at the 09:30 open that does not
happen, an early close at 13:00. A date published or moved after the cutoff is not here.

Events of each type before and after the snapshot's last day, and the other types released at the
same minute, whose effects cannot be told apart from it there:

| event | before cutoff | after cutoff | at the same time as |
|---|---|---|---|
| CPI | 13 | 1 | — |
| EARNINGS | 18 | 0 | — |
| EIA_PETROLEUM | 62 | 3 | — |
| FOMC | 9 | 2 | — |
| GDP | 13 | 1 | PCE 12/14 |
| NFP | 14 | 0 | RETAIL_SALES 2/14 |
| NYSE_CLOSED | 11 | 4 | — |
| NYSE_EARLY_CLOSE | 2 | 2 | — |
| PCE | 14 | 1 | GDP 12/15 |
| PPI | 13 | 1 | RETAIL_SALES 3/14 |
| RETAIL_SALES | 15 | 1 | NFP 2/16, PPI 3/16 |
