# Suggestions & Out-of-Scope Observations (`campaign-7` / `gemini-4-carbon`)

Per `CAMPAIGN.md`, this document records data nuances, evaluation-harness bottlenecks, and calendar/infrastructure improvements observed during the campaign that fall outside the scope of `agent/modeling.py`.

*(Environment integrity note per Rule 6: No operator miner models or proprietary materials were encountered anywhere in the workspace or environment.)*

---

## 1. Documenting Exact Bar-Label vs Validator Timestamp Alignment Around Scheduled Releases

In `market_data/prices/{ASSET}/1m/`, each minute bar labeled `T` (e.g., `12:30:00Z`) aggregates trades over `[T, T + 60s)`—so `close[12:30:00Z]` already contains the full first minute of price discovery following a `12:30:00Z` (`08:30:00 EDT`) `CPI` or `NFP` release. Meanwhile:
- `synth_lib.benchmark.generate_predictions.context_slice()` strictly slices `index < start_time` (ending at `12:29:00Z` for a `12:30:00Z` prompt—strictly leak-free), whereas
- Validator scoring aligns realized step targets at `start_time + k * time_increment` directly against bar label `T`. Consequently, for an on-the-hour `CRYPTO_24H` prompt (`time_increment = 300s`), the `12:30:00Z` release bar lands at the **endpoint of step `[12:25, 12:30]`**, not inside step `[12:30, 12:35]`.

Adding a 4-line diagram in `DATA.md` illustrating how open-time minute bar labels map onto 5-minute scoring steps around `events.parquet` timestamps would save miners and future research agents from a subtle off-by-one 5-minute event-alignment bug (fixing this alignment in Iteration 4 was critical to making `24H` macro event kernels improve CRPS).

---

## 2. Native Multi-Competition & `--fast-crps` Mode in Starter `agent/predict.py`

In the starter `agent/predict.py`, evaluating a single 10-day window across all 18 competition-asset cells requires 18 separate Python invocations and calls `backtest()`, which recomputes ~2,850 historical moving-average leaderboard rounds across ~256 miners per asset (~90–105 seconds per `CRYPTO_1H` asset on 14 vCPUs, even though `CAMPAIGN.md` rightly emphasizes judging strictly on CRPS rather than field rank).

Upgrading the starter `agent/predict.py` with:
1. `--competition {CRYPTO_24H, COM_EQU_24H, CRYPTO_1H, ALL}` sharing a single `ProcessPoolExecutor` across `synth-lib`'s official scoring functions,
2. `--fast-crps`, which evaluates the exact validator `prompt_times` from the offline score bundle via `synth_lib.backtester.orchestration.score_crps()` (`0.0` floating-point discrepancy vs `backtest()`), and
3. `--seed` forwarding to `SYNTH_BENCHMARK_SEED` for paired common-random-numbers comparisons

reduced a 3-window, 54-cell, 1000-path evaluation from ~48 minutes to ~7.5 minutes (`6.4x` wall-clock speedup) with zero loss of exactness. Shipping these flags in the starter `predict.py` would let agents spend significantly more of their wall-clock envelope on hypothesis testing rather than reward-table replay.

---

## 3. Documenting Closed-Weekend Perpetual Microstructure in `DATA.md`

`DATA.md` helpfully notes that flat candles (`trade_count == 0`) occur outside active hours and that weekday volatility runs 2–4x hotter than weekends on `COM_EQU_24H`. Two additional empirical properties of Hyperliquid equity/commodity perpetuals during weekend underlying closures (`Fri 17:00 NY -> Sun 18:00 NY`) are worth highlighting for miners:
1. **Strong Sub-Diffusive Mean Reversion (`VR < 1`)**: Because perpetual mark/funding mechanics tether weekend perp prices toward Friday's cash/Globex settlement while the underlying exchange is shut, 24-hour weekend variance is `30%–55%` smaller than the sum of 288 5-minute variances (`wknd_ou_kappa ~ 0.015–0.025` per 5m step). Scaling 1-minute variance by `5x` or simulating a weekend random walk systematically over-predicts 3h–24h CRPS intervals.
2. **Discrete Flat-Bar Mass on 5-Minute Intervals**: Even at 5-minute resolution, closed-Saturday equity/commodity perpetuals produce exact zero returns (`r_5m == 0`) on `22%–55%` of bars, creating a mixed continuous + point-mass return distribution at short scoring lags (`5m, 30m`).

---

## 4. High-Value Additions to `$SYNTH_BENCHMARK_CALENDAR` (`events.parquet`)

The daily-refreshed `events.parquet` feed proved highly effective once validated via Leave-One-Event-Out (LOEO) QLIKE (`FOMC`, `CPI`, `NFP`, `PPI`, `PCE`, `EARNINGS`, and `NYSE_CLOSED` / `NYSE_EARLY_CLOSE`—including a notable `0.52x–0.68x` US cash-hour volatility drop on institutionalized crypto during NYSE holidays when US Spot Crypto ETFs are shut). Four scheduled additions would meaningfully reduce the remaining unexplained variance spikes:
1. **ISM Manufacturing & Services PMI (`10:00 NY`)** and **Weekly Initial Jobless Claims (`Thu 08:30 NY`)**: Responsible for several of the largest non-`CPI/NFP` weekday morning volatility jumps on `XAU`, `SP500`, and `BTC`.
2. **OPEC+ Ministerial Meetings**: `WTIOIL` was consistently the toughest `COM_EQU_24H` cell relative to top specialist miners (`EIA_PETROLEUM` at `10:30 NY` failed LOEO because weekly inventory noise is already absorbed by Wednesday morning seasonality, whereas weekend/scheduled OPEC+ production decisions drive large Sunday `18:00 NY` / Monday gaps).
3. **Top `SP500` Index-Weight Earnings (`MSFT`, `AMZN`, `META`, `AVGO`)**: Because `SP500` carries the single highest competition weight (`coef = 3.438` in `COM_EQU_24H`), tagging mega-cap constituent after-hours earnings (`16:00–16:15 NY`) with `scope = SP500` would capture predictable index-level post-close jumps even for tickers without standalone Hyperliquid perps.
