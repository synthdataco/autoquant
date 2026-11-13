# Calibrated price-path simulation for Synth

## Context

The objective was lower CRPS on all three current Synth configurations, not a better field-relative rank. The research used only the frozen minute-OHLCV snapshot through 2026-11-08, the supplied event calendar, and the official synth-lib backtester at 1000 paths. No online prices, post-cutoff scores, or other miners' served paths were used.

The scoring structure matters. Both 24h configurations score 288 five-minute returns, 48 30-minute returns, eight three-hour returns, and one terminal absolute price. Short-horizon marginal calibration therefore dominates. Crypto-1h scores several price-change horizons plus realized-volatility CRPS on 5-, 15-, and 60-minute blocks, so it also needs realistic within-path volatility.

## Approach

The final model is deliberately small, offline, and fast. It uses only the context passed to `simulate()` and the calendar named by `$SYNTH_BENCHMARK_CALENDAR`.

For the two 24h formats, it uses:

- A 14-day context, with invalid prices and returns across NaN gaps removed rather than interpolated.
- New York-time 30-minute volatility profiles, separately for weekdays and weekends. Profile variance is shrunk toward the full-context level and RMS-normalized.
- A 50/50 mixture of Gaussian innovations and empirical five-minute residuals. Residuals are median-centered and clipped at eight standard deviations to control repeated extreme jumps.
- Early-fit, localized multipliers for CPI, NFP, FOMC, and earnings. Weak or unstable PPI, retail-sales, GDP/PCE, and EIA effects are excluded.
- Antithetic paths: the second half of each pair is the negative of the first, reducing finite-ensemble mean error.

For crypto-1h, it instead uses:

- Only the last seven days of the 14-day input context.
- A 30-minute EWMA of squared one-minute returns for the current scale. The scale adapts faster than the 24h estimator, which materially improves this format.
- Median-centered empirical one-minute residuals sampled in 15-minute contiguous blocks. This preserves short-run volatility clustering and zero runs without the stale regime carried by 60-minute blocks.
- The same early-fit macro-event profiles, with each profile entry applied for five minutes.
- Antithetic sampling.

All random generators are seeded from `SYNTH_BENCHMARK_SEED`, with a fixed local fallback. The implementation honors the requested path count and output horizon and tolerates NaN-only or partially missing contexts.

## Experiments

The campaign changed one component at a time and used disjoint latest and older windows. Small effects needed consistent signs across asset-window cells; a latest-only gain was not enough.

The main positive findings were:

- Replacing the starter's full-context Gaussian scale with a 30-minute EWMA reduced latest BTC-1h CRPS by 12.80%. The same fast scale hurt both initial 24h test assets and was not used there.
- New York-time seasonality improved the three initial 24h assets by 3.38–5.46% in the latest window and passed paired older-window checks. Its benefit was especially large on tokenised equities.
- The Gaussian/empirical mixture improved all 13 assets in the September window and 7 of 13 in the latest window. Pure empirical residuals were less robust and were rejected.
- Fourteen days was approximately neutral on average across all latest 24h assets and improved 12 of 13 September assets. The final 1h path retains a seven-day slice to avoid a small older HYPE regression.
- Conservative calendar scaling improved nine of ten late 24h event cells and every tested late 1h NFP, CPI, and FOMC cell. Earnings produced the largest gains, 1.23% on TSLAX and 1.89% on GOOGLX in the selected windows.
- Fifteen-minute residual blocks improved all five latest crypto-1h assets and were neutral on average in October. Antithetic sampling improved five of six 24h test cells and eight of ten 1h cells.

Honest failures and rejected alternatives were also informative:

- A 60-minute half-life for 1h regressed in five of six test cells; 15-minute half-lives were not consistently better than 30.
- Applying the 24h intraday profile to 1h caused large regressions on every asset in both windows, generally 2–6%. The fast 1h EWMA already captures the useful information.
- Sixty-minute residual blocks improved the latest window but worsened October, so they were rejected. The shorter 15-minute blocks had the better robustness profile.
- A 30-day context produced a large older BTC gain but materially hurt latest XAU and TSLAX. It was rejected in favor of 14 days.
- EIA, retail sales, PPI, and collinear GDP/PCE calendar effects did not replicate well enough across early and late samples to include.
- No foundation model was used: the measured bottlenecks were conditional scale, residual shape, and path-level volatility, all addressed within the latency budget by transparent estimators.

## Results

Final numbers below are equal-asset mean CRPS from the official backtester at 1000 paths. The 24h runs cover the 10 scored days ending 2026-11-08; crypto-1h covers five scored days with the latest formula. Starter and champion use identical prompts and windows. Detailed per-asset values are in `journal.md`.

| Configuration | Starter CRPS | Champion CRPS | Change |
|---|---:|---:|---:|
| CRYPTO_24H | 4607.927 | 4494.773 | −2.46% |
| COM_EQU_24H | 3304.762 | 3119.334 | −5.61% |
| CRYPTO_1H | 979.258 | 942.752 | −3.73% |

All 18 assets improved individually. The runs covered 1200 CRYPTO_24H prompts, 870 COM_EQU_24H prompts, and 3600 CRYPTO_1H prompts, with 100% realized-price coverage in every starter and champion run.

The exact frozen file passed compilation, path-shape, positivity, finiteness, all-NaN fallback, and seed-reproducibility checks. Measured generation latency at 1000 paths was approximately 0.06 seconds for 24h and 0.02 seconds for 1h on the campaign machine, leaving ample headroom for live serving.

## Limitations

The model remains sensitive to volatility-regime changes. Empirical residuals assume the recent context is representative, and calendar multipliers are rounded estimates from a small number of releases, not a full event model. Several component gains are below 0.5% and were accepted only because their signs were consistent across assets and windows. Multiple research comparisons can still leave selection bias, so the paired final runs are important but are not a guarantee on the unseen ten days.

The model does not use cross-asset information, order-book state, external forecasts, or a learned drift. Its median-centered design is intentional: directional prediction was not supported strongly enough by the experiments. Only close prices drive calibration; range, volume, and trade-count features remain possible sources of incremental information.

## What to do with more budget

- Pre-register a regime-adaptive mixture weight or scale estimator and test it on more disjoint high- and low-volatility periods.
- Evaluate event multipliers with explicit leave-one-event-out fits and more release types, especially earnings, where the sample is small but effects are large.
- Test range, volume, and trade-count features only for the information they add beyond the fast EWMA.
- Compare deterministic quantile or low-discrepancy ensembles against antithetic sampling at 1000 paths.
- Run longer forward evaluations and test whether shorter block mixtures can further improve realized-volatility CRPS without preserving stale regimes.
