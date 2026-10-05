# Calendar volatility and price-fan calibration for Synth

## Context

Synth scores simulated price fans with the validator's CRPS. Lower CRPS is better. The current configurations are CRYPTO_24H, COM_EQU_24H, and CRYPTO_1H; the last also scores realized-volatility blocks. All results here use the frozen venue-matched minute snapshot and the official `synth_lib.backtester` scorer. The data cutoff is 2026-09-14.

## Approach

The starter was a Gaussian log-price random walk with one volatility estimate from the preceding seven days. I fit a per-asset relative minute-variance profile by UTC hour and weekday/weekend using adjacent valid closes through 2026-07-31. Weekly normalization removes broad volatility-regime changes before averaging calendar cells; neighboring hours are smoothed and the result is shrunk toward each weekday/weekend mean. At each prompt, the model estimates the current variance level from the preceding seven-day OHLCV frame after dividing valid adjacent squared log returns by the fitted calendar factor. It projects that factor across every future minute, then sums minute variances within the requested output step. Crypto also blends the seven-day variance level equally with a 24-hour-half-life estimate. Simulated log returns have zero drift; every path starts at the last valid context close.

Simulated step returns use unit-variance Student-t(5) shocks for every 24-hour format and for HYPE 1-hour. COM_EQU 24-hour fans pair each shock path with its negative to reduce finite-sample imbalance. ETH, SOL, and XRP 1-hour use Student-t(3); BTC 1-hour uses Student-t(5) with a pathwise lognormal volatility multiplier. On crypto 1-hour weekend prompts, a July-trained regression of recent 1-hour, 6-hour, 24-hour, and 7-day adjusted variance replaces the usual volatility level. HYPE 1-hour uses that regression on weekdays as well. Crypto 24-hour applies a 0.70 factor to predicted variance for weekend starts and 0.85 for weekday starts. The eight COM_EQU assets use a calendar profile fitted through August 31: 5-minute weekday bins and the original hourly weekend bins. Crypto retains the July hourly profile. The model respects `time_increment`, `time_length`, and `num_simulations`. It handles NaN minutes without interpolation and seeds its NumPy generator from `SYNTH_BENCHMARK_SEED` and the prompt identity. The serialized profiles and regression coefficients are included in the repository, so serving needs no network access.

## Experiments

The first evaluator used a regular hourly prompt grid. Historical prompt starts actually have irregular minute offsets, and synth-lib's matching tolerance can pair a prompt with a later prediction. Those early CRPS figures were discarded. The corrected evaluator queries the offline score cache for exact prompt starts and checks snapshot coverage through the backtester's 26-hour price-read buffer. A separate implementation bug compared raw timestamp integers as if they were nanoseconds; on the pandas timestamp unit in this environment, it excluded every adjacent return and made the fan far too wide. The journal records both failed attempts and their scores.

On two scored-date windows, August 15–16 and September 9–10, every active COM_EQU asset improved with fixed calendar seasonality relative to the starter. The average per-asset CRPS moved from 1613.31 to 1260.84 on the August weekend and from 2523.42 to 2411.60 in September. Across five crypto assets, the starter, fixed-calendar, and recent-volatility models averaged 1697.79, 1582.61, and 1572.80 for 24-hour scoring in August; 3862.87, 3847.72, and 3831.82 in September. For 1-hour scoring the corresponding figures were 349.90, 319.37, and 315.47 in August; 873.12, 868.41, and 866.21 in September. Each asset had 48 prompts per 24-hour format or 288 per 1-hour format, with 1000 paths and full realized-price coverage.

The recent-volatility effect is small next to the calendar effect. It improved fixed seasonality for every crypto 1-hour asset in both windows, while BTC 24-hour on the August weekend regressed by 0.5%. The working rule uses the recent blend for crypto and fixed calendar variance for COM_EQU. Standardized one-minute returns had heavy tails in the January–July fit period: their 99.9th absolute percentiles were 6.39 for BTC, 6.69 for ETH, and 7.80 for XAU, versus 3.29 for a standard normal. Student-t(5) shocks improved all 18 asset-format pairs over Gaussian shocks in the ten-day September test.

Heavier Student-t(3) shocks improved the quiet August weekend, but raised the September 9–10 COM_EQU CRPS for every sampled asset and slightly raised the five-asset crypto 1-hour mean. Independent checks supported Student-t(3) for ETH, SOL, and XRP 1-hour only. A later Student-t(2.5) test raised their September 1–2 mean CRPS from 919.99 to 924.62; all three assets regressed, so it was discarded. Student-t(4) improved the August BTC/XAU sample but its September gain was only about 0.05%, below the predeclared expansion threshold. A four-window regression using past 1-hour, 6-hour, 24-hour, and 7-day variance lowered August 1-hour CRPS, then raised the September weekday mean; the all-week version was discarded. Applying it only on weekends improved five-asset 1-hour CRPS on both August 20–29 and September 1–8. Combined with the selected 1-hour shock laws, it improved the September 3–12 mean from 803.14 to 799.89. Refitting the COM_EQU calendar through August 31 improved all eight assets on September 1–8 and September 3–12.

A final test multiplied the predicted variance by 0.85. This improved all five crypto 24-hour assets on the independent August 5–14 holdout and lowered their mean by 0.31%; on September 3–12 it lowered the mean by 0.28%, with a 0.02% HYPE regression. The same factor raised crypto 1-hour CRPS on September 9–10 by 0.16% and COM_EQU 24-hour CRPS by 0.13%, so those formats retain their earlier variance rules. A stronger 0.70 crypto 24-hour factor improved the August weekend but raised September 9–10 mean CRPS by 0.27%, so it was rejected.

Applying 0.70 only on weekend crypto 24-hour starts improved three mixed windows over 0.85 throughout: September 3–12 by 0.05%, August 5–14 by 0.04%, and the independent August 20–29 window by 0.23%. All five assets improved on the last window, so the conditional rule replaced the constant 0.85 factor.

Two further 1-hour tests failed their short-window gates. Widening weekday variance by 10% raised the five-asset September 1–2 mean by 0.20%, with a 0.61% HYPE regression. Giving HYPE the pathwise volatility multiplier used for BTC slightly improved the August weekend but raised its September 9–10 CRPS by 0.23%.

Adding the crypto-style recent variance blend to COM_EQU improved its September 1–2 mean by 0.62%, mostly through NVDAX, but raised XAU's September 5–6 weekend CRPS by 1.51%. The configuration-wide blend failed its predeclared asset limit and was discarded.
Restricting that blend to NVDAX also failed: its September 1–2 gain did not persist on September 5–6, where CRPS rose by 0.14%. Limiting the NVDAX blend to weekday starts raised its September 9–10 CRPS by 0.027%, so that variant was also discarded.

The selected hourly COM_EQU profile smoothed the cash opening spike across neighboring hours. Fitting 30-minute weekday bins through August 31, while preserving the old weekday mean and weekend factors, placed the peak at 13:30 UTC for seven assets and 14:00 for SP500. The eight-asset mean improved by 0.68% on September 1–2, 0.28% on September 5–6, 0.30% on September 9–10, and 0.32% on September 3–12.
Fitting 15-minute weekday bins sharpened that opening profile further. Relative to the 30-minute version, it improved all eight assets on both two-day weekday screens and the weekend screen, and lowered the September 3–12 mean by 0.16%; WTIOIL alone rose by 0.05% on the ten-day window.

Fitting 5-minute weekday bins lowered the COM_EQU mean on September 1–2, September 5–6, September 9–10, and September 3–12 by 0.31%, 0.18%, 0.17%, and 0.18%, respectively, relative to 15-minute bins. Every asset improved on the full window, with the smallest gain of 0.04% for WTIOIL.

Regularized day-of-week COM_EQU factors, fit through August 31, lowered the September 1–2 and September 5–6 means by 0.075% and 0.087% but raised the September 9–10 mean by 0.019%. That inconsistency failed the predeclared rule, so the day factors were discarded.

A separate July-trained 24-hour HAR variance forecast raised the independent August 5–14 crypto mean from 2501.46 to 2512.13; all five assets regressed. The selected seven-day/recent blend remains the day-ahead variance rule.

Capping the top 0.1% of COM_EQU context squared returns improved the September 1–2 and September 5–6 means by 0.31% and 0.37%, but raised the September 9–10 mean by 0.05%. This failed the predeclared consistency rule; the raw seven-day adjusted mean was retained.

Pairing COM_EQU Student-t(5) shock paths antithetically lowered mean CRPS on September 1–2, September 5–6, September 9–10, and September 3–12 by 0.036%, 0.078%, 0.027%, and 0.043%, respectively. The ten-day mean moved from 1940.77 to 1939.94, and seven of eight assets improved. Two additional weekday comparisons under an alternate benchmark seed also improved, by 0.043% and 0.031%. Direct serving backtests reproduced all eight candidate scores exactly.

The same path pairing in crypto 24-hour improved the five-asset mean by 0.018% on an independent August 5–14 holdout, 0.035% on September 9–10, and 0.034% on September 3–12. An alternate-seed robustness check was stopped when the harness issued its landing order, so this small-gain candidate was not added to the selected serving model.

An asset-specific revision to the earlier all-week HAR failure succeeded: using the hourly variance regression on HYPE weekdays lowered HYPE 1-hour CRPS by 0.59% on September 1–8, 1.33% on August 20–29, and 0.42% on September 3–12. The other four crypto 1-hour assets keep the weekend-only rule.

## Results

The official September 3–12 backtest used 1000 paths per prompt, 240 exact starts per 24-hour asset, 1440 per 1-hour asset, and full realized coverage. The table shows the unweighted mean of each asset's prompt-level CRPS within its configuration. Lower is better.

| Configuration | Starter | Calendar Gaussian | Student-t(5) | Selected model |
|---|---:|---:|---:|---:|
| COM_EQU_24H (8 assets) | 2176.07 | 1981.98 | 1962.43 | 1939.94 |
| CRYPTO_24H (5 assets) | 3903.16 | 3824.47 | 3797.67 | 3785.15 |
| CRYPTO_1H (5 assets) | 831.11 | 810.84 | 806.57 | 799.00 |

All 18 Student-t(5) asset-format CRPS values were below their matched Gaussian and starter values. The initial selected module produced all 18 expected asset-format results with exact prompt counts and 100% realized coverage. A crypto 24-hour variance adjustment passed a separate August 5–14 holdout (2510.26 → 2502.36) and the September 3–12 confirmation (3797.67 → 3787.20). The selected weekend refinement further lowered September 3–12 crypto 24-hour CRPS to 3785.15 and passed an independent August 20–29 confirmation. Direct serving-module backtests reproduced all five final crypto 24-hour candidate scores exactly, with 240 prompts and full coverage per asset. A uniform adjustment to crypto 1-hour and COM_EQU was rejected after September 9–10 regressions; HYPE's weekday HAR exception improved on two longer windows, and its direct serving backtest reproduced the candidate's 1054.19 CRPS. The finer COM_EQU weekday profiles passed three short screens and the full September confirmation; direct serving backtests reproduced all eight five-minute scores and all eight subsequent antithetic scores exactly, with 240 prompts and full coverage per asset. The selected model's gain over the starter is 10.85% for COM_EQU_24H, 3.02% for CRYPTO_24H, and 3.86% for CRYPTO_1H.

## Limitations

The July calendar profile preserves forward validation on both August and September windows; the COM_EQU refit through August 31 has only September validation. Newer listings have shorter, thinner training histories. The model uses closes for its variance estimate and does not model drift or cross-asset information. Only BTC 1-hour has explicit pathwise volatility uncertainty. The short-window tests include different volatility regimes, and a small percentage gain may not persist. The selected antithetic COM_EQU gain is 0.043% on the ten-day validation window; its direction held on two alternate-seed weekday checks, but those checks do not prove a future gain. Every backtest is before the September 14 data cutoff; the ten days after that cutoff remain unseen.

## With more budget

I would test a richer state-dependent volatility forecast on additional forward windows, fit the calendar surface with recency weights once enough history accumulates for newer listings, and compare a direct CRPS-trained forecaster against the simple seven-day/24-hour blend. Exact prompt matching and stored per-prompt CRPS would make those comparisons more reliable.
