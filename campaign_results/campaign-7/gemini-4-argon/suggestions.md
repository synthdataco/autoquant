# Out-of-Scope Suggestions for the Synth Benchmark & Harness

During our research campaign, we identified several concrete observations and suggestions for the benchmark harness and calendar pipeline that fall outside the scope of `agent/modeling.py`:

## 1. Include Pre-Market vs. After-Hours Session Flag (`BMO` vs. `AMC`) in `events.parquet`
- **Observation**: In `$SYNTH_BENCHMARK_CALENDAR` (`events.parquet`), corporate `EARNINGS` events for the current five equities (`NVDAX`, `TSLAX`, `GOOGLX`, `AAPLX`, `SPCX`) are all After-Market-Close (`AMC`, stamped at `16:00 America/New_York`).
- **Suggestion**: If future equity additions to `COM_EQU_24H` report earnings Before Market Open (`BMO`, e.g., `07:00–08:30 America/New_York`) rather than After Market Close (`AMC`), adding an explicit `session` metadata column (`BMO` vs. `AMC`) or release-window duration in `events.parquet` would allow event response curves to distinguish pre-open vs. post-close releases without relying solely on hour-of-day heuristics.

## 2. Historical Validator Leaderboard (`CRYPTO_1H`) Interval-Set Cutover Helper
- **Observation**: As documented in `CAMPAIGN.md` and `synth_lib/configs.py`, the live SN50 subnet updated the `CRYPTO_1H` scoring interval set (`HIGH_FREQUENCY`) on `2026-09-22T09:00:00Z`. When running `agent/predict.py --latest-formula` on windows before `2026-09-22` (such as `R1_Aug` and `R2_Sep`), CRPS is evaluated consistently under the current formula, while historical field-relative ranks are omitted (`rank = None`) because the archived field scores before `2026-09-22` used the legacy formula.
- **Suggestion**: Pre-computing and bundling a reference baseline or synthetic field quantile table under the post-`2026-09-22` `CRYPTO_1H` formula for August–September 2026 in the offline backtester bundle would provide a convenient field-percentile reference across older 1H regimes alongside raw CRPS.

## 3. Add Scheduled Crypto Network Upgrade / Token Unlock Events to `events.parquet`
- **Observation**: While `events.parquet` provides comprehensive macro (`FOMC`, `CPI`, `PPI`, `NFP`, `PCE`, `GDP`, `RETAIL_SALES`), commodity (`EIA_PETROLEUM`), exchange holiday (`NYSE_CLOSED`, `NYSE_EARLY_CLOSE`), and equity `EARNINGS` timestamps, crypto-native scheduled events (e.g., major scheduled token unlocks for `HYPE`/`SOL`/`XRP` or major network upgrades) are not currently represented.
- **Suggestion**: Adding deterministic, publicly scheduled token unlock dates or major protocol upgrade timestamps to `events.parquet` could extend scheduled-event calibration to `CRYPTO_24H` and `CRYPTO_1H`.
