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
| AAPLX | 2025-11-21 | 2026-09-14 | 298 | 42.4% | 60.1% | 18.4% |
| BTC | 2025-08-01 | 2026-09-14 | 410 | 0.0% | 0.0% | 0.0% |
| ETH | 2025-08-01 | 2026-09-14 | 410 | 0.0% | 0.0% | 0.0% |
| GOOGLX | 2025-11-18 | 2026-09-14 | 301 | 25.2% | 23.9% | 13.4% |
| HYPE | 2025-08-01 | 2026-09-14 | 410 | 0.1% | 0.1% | 0.0% |
| NVDAX | 2025-11-12 | 2026-09-14 | 307 | 10.6% | 14.9% | 3.3% |
| SOL | 2025-08-01 | 2026-09-14 | 410 | 0.0% | 0.0% | 0.0% |
| SP500 | 2026-03-18 | 2026-09-14 | 181 | 1.0% | 2.8% | 0.6% |
| SPCX | 2026-05-17 | 2026-09-14 | 121 | 2.5% | 4.6% | 3.8% |
| TSLAX | 2025-11-13 | 2026-09-14 | 306 | 14.7% | 26.2% | 12.7% |
| WTIOIL | 2026-01-06 | 2026-09-14 | 252 | 12.5% | 58.8% | 2.4% |
| XAU | 2025-12-22 | 2026-09-14 | 267 | 11.0% | 39.2% | 4.5% |
| XRP | 2025-08-01 | 2026-09-14 | 410 | 0.0% | 0.0% | 0.0% |

**Cached realized paths** (13 assets): **2026-06-20 → 2026-09-14**. Where a
prompt's slice of the price store contains NaN, scoring substitutes the validator's own
array for that prompt. Outside this range it cannot, and the prompt is scored on the
points that remain — which is a smaller CRPS from a thinner reference, not a better model.
